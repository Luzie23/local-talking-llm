"""
APP — MAIN ENTRY POINT
========================
This is the file you run to start the assistant. It does NOT contain the
"real" logic itself — it just reads the command-line options, sets up the
web server (or the terminal loop), and calls out to the other files in this
folder:

  - llm_provider.py  -> generates the assistant's text reply
  - stt.py           -> turns speech into text (voice condition only)
  - tts.py           -> turns text into speech (voice condition only)
  - speech.py        -> recording/playback/audio-format glue (voice condition only)

How to run it:
    python backend/app.py            (terminal mode, asks for microphone input)
    python backend/app.py --web      (starts a local web server, open in browser)

See the project's README for the full list of command-line options
(model choice, voice cloning, language, etc.).
"""
import argparse
import os
import threading
import time
from queue import Queue

import numpy as np
from flask import Flask, jsonify, request

import llm_provider
import speech
import stt
from tts import TextToSpeechService

# --- Command-line options -------------------------------------------------
parser = argparse.ArgumentParser(description="Local Voice Assistant with ChatterBox TTS")
parser.add_argument("--voice", type=str, help="Path to voice sample for cloning")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemma3 for ollama, MiniMax-M2.7 for minimax)")
parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "minimax"],
                    help="LLM provider: 'ollama' for local models, 'minimax' for MiniMax cloud API (default: ollama)")
parser.add_argument("--api-key", type=str, default=None, help="API key for cloud LLM providers (or set MINIMAX_API_KEY env var)")
parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--web", action="store_true", help="Run the app as a Flask web server instead of terminal mode")
parser.add_argument("--language", type=str, default="de", help="Language code for speech transcription (default: de)")
args = parser.parse_args()

# --- Set up the language model and the voice engine -----------------------
llm = llm_provider.create_llm(
    provider=args.provider,
    model=args.model,
    api_key=args.api_key,
    temperature=args.temperature,
)
chain_with_history = llm_provider.build_chain(llm)
tts = TextToSpeechService()

# --- Web server (used with --web) ------------------------------------------
# The frontend/ folder (the web page the participant sees) lives one level up
# from this file, so we point Flask there explicitly.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.route("/")
def index():
    """Serve the web interface's main HTML page."""
    return app.send_static_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Text-condition endpoint: the participant typed a message in the browser.
    Expects JSON: {"text": "..."}.
    Returns JSON: {"response": "..."}.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Kein Text eingegeben."}), 400

    response = llm_provider.get_llm_response(chain_with_history, text)
    return jsonify({"response": response})


@app.route("/api/chat/audio", methods=["POST"])
def chat_audio_endpoint():
    """
    Voice-condition endpoint: the participant spoke into the browser's microphone.
    Expects multipart/form-data with an "audio" file and optional "language" field.
    Returns JSON: {"transcript": "...", "response": "...", "audio": "...base64..."}.
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file uploaded."}), 400

    language = request.form.get("language") or args.language
    try:
        audio_np = speech.decode_uploaded_wav(audio_file.read())
    except Exception as e:
        return jsonify({"error": f"Unable to decode audio: {e}"}), 400

    transcript = stt.transcribe(audio_np, language=language)
    response = llm_provider.get_llm_response(chain_with_history, transcript)

    sample_rate, audio_array = tts.long_form_synthesize(
        response,
        audio_prompt_path=args.voice,
        exaggeration=speech.analyze_emotion(response),
        cfg_weight=args.cfg_weight,
    )
    audio_base64 = speech.encode_wav_base64(sample_rate, audio_array)

    return jsonify({
        "transcript": transcript,
        "response": response,
        "audio": audio_base64,
    })


# --- Terminal mode (used without --web) -------------------------------------
def run_terminal_mode():
    """
    A simple command-line loop: press Enter to record, speak, press Enter
    again to stop, and the assistant replies out loud. Useful for quick local
    testing without opening a browser.
    """
    print("\U0001F916 Local Voice Assistant with ChatterBox TTS")
    print("\u2500" * 42)
    selected_language = input("Which language should I use for transcription? (e.g. de, en, fr): ").strip().lower() or "en"

    if args.voice:
        print(f"Using voice cloning from: {args.voice}")
    else:
        print("Using default voice (no cloning)")

    print(f"Emotion exaggeration: {args.exaggeration}")
    print(f"CFG weight: {args.cfg_weight}")
    print(f"LLM model: {args.model or ('gemma3' if args.provider == 'ollama' else 'MiniMax-M2.7')}")
    print(f"LLM provider: {args.provider}")
    print("\u2500" * 42)
    print("Press Ctrl+C to exit.\n")

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    try:
        while True:
            input("\U0001F3A4 Press Enter to start recording, then press Enter again to stop.")

            data_queue: Queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=speech.record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size == 0:
                print("No audio recorded. Please ensure your microphone is working.")
                continue

            print("Transcribing...")
            text = stt.transcribe(audio_np, language=selected_language)
            print(f"You: {text}")

            print("Generating response...")
            response = llm_provider.get_llm_response(chain_with_history, text)

            dynamic_exaggeration = speech.analyze_emotion(response)
            # More emotional replies are spoken with a slightly lower cfg_weight,
            # which makes the delivery a bit more expressive/less flat.
            dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

            sample_rate, audio_array = tts.long_form_synthesize(
                response,
                audio_prompt_path=args.voice,
                exaggeration=dynamic_exaggeration,
                cfg_weight=dynamic_cfg,
            )

            print(f"Assistant: {response}")
            print(f"(Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})")

            if args.save_voice:
                response_count += 1
                filename = f"voices/response_{response_count:03d}.wav"
                tts.save_voice_sample(response, filename, args.voice)
                print(f"Voice saved to: {filename}")

            speech.play_audio(sample_rate, audio_array)

    except KeyboardInterrupt:
        print("\nExiting...")

    print("Session ended. Thank you for using the assistant!")


if __name__ == "__main__":
    if args.web:
        print("\U0001F916 Starting Flask web server for Local Voice Assistant")
        print("Open http://127.0.0.1:5000 in your browser to connect.")
        app.run(host="127.0.0.1", port=5000)
    else:
        run_terminal_mode()
