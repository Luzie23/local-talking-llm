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
(model choice, language, etc.), and for how to download a Piper voice
before running the voice condition for the first time.
"""
import argparse
import json
import os
import re
import threading
import time
from queue import Queue

import numpy as np
from flask import Flask, Response, jsonify, request

import llm_provider
import speech
import stt
import study_settings
from behavior_profiles import load_behavior_profile
from interview_guide import InterviewGuide
from tts import TextToSpeechService

# --- Load study settings (see study_content/study_settings.yaml) ---------
# Loaded before the command-line options below, so that e.g. --language has
# a sensible default coming from the researcher-editable settings file
# instead of being hardcoded here.
_settings = study_settings.load_study_settings()

# --- Command-line options -------------------------------------------------
parser = argparse.ArgumentParser(description="Local Voice Assistant")
parser.add_argument("--model", type=str, default=None, help="LLM model name (default: gemma3 for ollama, MiniMax-M2.7 for minimax)")
parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "minimax"],
                    help="LLM provider: 'ollama' for local models, 'minimax' for MiniMax cloud API (default: ollama)")
parser.add_argument("--api-key", type=str, default=None, help="API key for cloud LLM providers (or set MINIMAX_API_KEY env var)")
parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature (default: 0.7)")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples to voices/")
parser.add_argument("--web", action="store_true", help="Run the app as a Flask web server instead of terminal mode")
parser.add_argument("--language", type=str, default=_settings.get("language", "de"),
                    help="Language code for speech transcription (default: from study_settings.yaml, else 'de')")
args = parser.parse_args()

# --- Set up the language model and the voice engine -----------------------
llm = llm_provider.create_llm(
    provider=args.provider,
    model=args.model,
    api_key=args.api_key,
    temperature=args.temperature,
)
chain_with_history = llm_provider.build_chain(llm)

# Which Piper voice to use is a researcher setting (study_content/study_settings.yaml),
# not a command-line option — see backend/tts.py's module docstring for how
# to download a voice the first time.
_tts_voice = _settings.get("tts_voice", "de_DE-thorsten-high")
tts = TextToSpeechService(voice_name=_tts_voice)

# --- Load the active interview guide (see study_content/interviews/) ------
# Which guide file is active is controlled by study_content/study_settings.yaml,
# not by anything here — the fallback name below is only used if that
# setting is missing entirely.
_guide_filename = _settings.get("interview_guide", "depression_scid_example.yaml")
interview = InterviewGuide(_guide_filename)

# --- Load the behavior profile the guide asks for (see study_content/prompts/) ---
# This decides both the tone/strictness instructions and how the next fixed
# question is delivered (verbatim vs. naturally reworded) — see Step 3 of
# the project plan and backend/behavior_profiles.py.
profile = load_behavior_profile(interview.behavior_profile)

# --- Web server (used with --web) ------------------------------------------
# The frontend/ folder (the web page the participant sees) lives one level up
# from this file, so we point Flask there explicitly.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.route("/")
def index():
    """Serve the web interface's main HTML page."""
    return app.send_static_file("index.html")


def _split_sentences(text: str) -> list[str]:
    """
    Split a piece of text into individual sentences, so each one can be
    streamed to the browser (and spoken) as soon as it's ready, instead of
    waiting for the whole piece. This matters because "the feedback" or "a
    question" are not guaranteed to always be exactly one sentence — if a
    prompt in study_content/prompts/ is edited later to allow a longer,
    multi-sentence feedback, this still splits it up correctly instead of
    silently becoming one large, slower chunk.

    This is a simple, punctuation-based split (not aware of abbreviations
    like "Dr." or "z.B.") — good enough for short interview responses. Piper
    does its own, more careful sentence splitting internally for the actual
    audio; this split only decides how text is grouped for streaming.
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _tag_sentences(text: str, kind: str) -> list[dict]:
    """
    Like _split_sentences(), but each sentence is paired with a "kind"
    label ("feedback", "question", or "closing"). This is what lets the
    browser's "Frage wiederholen" (repeat question) feature replay only
    the actual question, not the feedback that came before it — see
    frontend/conversation.js's replayLastQuestion() and readStreamedReply().
    """
    return [{"text": s, "kind": kind} for s in _split_sentences(text)]


def _response_pieces(feedback: str, next_question: dict | None) -> list[dict]:
    """
    Break the reply that follows a piece of feedback into the individual
    sentences it's made of, in speaking order — this is what lets the voice
    condition start speaking the first sentence while later ones are still
    being synthesized (see Step 5b of the project plan). Each piece is
    tagged with its "kind" (see _tag_sentences()).

    For "natural_transition" delivery, the model blends the feedback and
    the next question into one smoothly-worded message — there are no
    separately-known pieces in that case, so the whole blended result is
    tagged "question" (the closest available approximation; "Frage
    wiederholen" will replay the blended feedback+question together for
    this delivery style, since they can no longer be told apart).
    """
    following_text = next_question["text"] if next_question is not None else interview.closing_message
    following_kind = "question" if next_question is not None else "closing"

    if profile.question_delivery == "natural_transition":
        combined = llm_provider.get_natural_transition(llm, feedback, following_text, language_code=args.language)
        return _tag_sentences(combined, "question")

    return _tag_sentences(feedback, "feedback") + _tag_sentences(following_text, following_kind)


def process_answer(answer_text: str) -> list[dict]:
    """
    The core interview turn: given what the participant just said, decide
    what the assistant says next, and advance the interview guide's state
    accordingly. Shared by both web routes and the terminal mode, so the
    behavior is identical regardless of modality.

    Returns a list of {"text": ..., "kind": ...} dicts, in speaking order,
    rather than one combined string — this is what backend/app.py's audio
    endpoint streams to the browser one at a time (see Step 5b of the
    project plan), and "kind" ("feedback" vs "question"/"closing") is what
    lets the browser's "Frage wiederholen" replay only the question.
    Callers that just want the full text (text condition, terminal mode)
    can join the "text" fields with spaces.

    Two cases:
      1. The participant is answering a follow-up that was asked last turn
         (interview.is_awaiting_followup_answer()) — no further follow-up
         is allowed here (capped at one per question), then the guide moves on.
      2. The participant is answering the current fixed question — a
         follow-up may be offered if this item allows it and none has been
         used for it yet.

    Prints a short debug line to the console for every turn (visible only
    in the terminal running this app, never sent to the participant) — this
    is meant to make it easy to check whether the follow-up mechanism is
    actually engaging as expected while testing, instead of having to guess
    from the final combined text alone.
    """
    if interview.is_awaiting_followup_answer():
        print(f"[interview] answering follow-up: \"{interview.pending_followup_text}\"")
        feedback, _ = llm_provider.get_feedback_and_followup(
            chain_with_history, profile, interview.pending_followup_text, answer_text,
            followup_allowed=False, language_code=args.language,
        )
        print(f"[interview] feedback={feedback!r}")
        next_question = interview.advance()
        return _response_pieces(feedback, next_question)

    question = interview.current_question()
    if question is None:
        # The guide was already finished; nothing left to ask.
        return _tag_sentences(interview.closing_message, "closing")

    followup_allowed = question["allow_followup"] and not interview.followup_already_used()
    print(
        f"[interview] question={question['id']!r} "
        f"allow_followup(guide)={question['allow_followup']} "
        f"followup_allowed(this turn)={followup_allowed}"
    )
    feedback, followup = llm_provider.get_feedback_and_followup(
        chain_with_history, profile, question["text"], answer_text,
        followup_allowed=followup_allowed, language_code=args.language,
    )
    print(f"[interview] feedback={feedback!r} followup={followup!r}")

    if followup:
        interview.start_followup(followup)
        return _tag_sentences(feedback, "feedback") + _tag_sentences(followup, "question")

    next_question = interview.advance()
    return _response_pieces(feedback, next_question)


@app.route("/api/interview/start", methods=["GET"])
def interview_start_endpoint():
    """
    Called once when the participant opens the page: returns the framing
    message and the first fixed question, verbatim from the active
    interview guide. No LLM call is involved — there is nothing to
    generate yet, only fixed text to present.
    Returns JSON: {"message": "..."}.
    """
    interview.reset()
    first_question = interview.current_question()
    if first_question is None:
        # An empty guide (e.g. a guide file with no questions yet).
        message = interview.framing_message
    else:
        message = f"{interview.framing_message} {first_question['text']}".strip()
    return jsonify({"message": message})


@app.route("/api/interview/start/audio", methods=["GET"])
def interview_start_audio_endpoint():
    """
    Voice-condition only: pre-synthesizes the opening message (framing +
    first question) as audio, sentence by sentence. Called in the
    background while the start overlay is still showing (see
    frontend/conversation.js), so spoken playback can begin immediately
    once "Interview starten" is clicked, instead of only starting to
    synthesize at that point.

    Unlike /api/chat/audio, this is returned as one plain JSON object, not
    a stream — the whole point of fetching it in the background is that it
    is ready and waiting well before it's actually needed, so there's no
    time-to-first-sound pressure here the way there is for a live reply.

    Returns JSON: {"pieces": [{"text": "...", "audio": "...base64...", "kind": "..."}, ...]}.
    "kind" is "framing" or "question" — lets the browser's "Frage
    wiederholen" repeat just the question, not the greeting, consistent
    with how later turns are tagged (see process_answer()).
    """
    interview.reset()
    first_question = interview.current_question()

    tagged_sentences = _tag_sentences(interview.framing_message, "framing")
    if first_question is not None:
        tagged_sentences += _tag_sentences(first_question["text"], "question")

    pieces = []
    for sentence in tagged_sentences:
        sample_rate, audio_array = tts.synthesize(sentence["text"])
        audio_base64 = speech.encode_wav_base64(sample_rate, audio_array)
        pieces.append({"text": sentence["text"], "audio": audio_base64, "kind": sentence["kind"]})

    return jsonify({"pieces": pieces})


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    Text-condition endpoint: the participant typed an answer in the browser.
    Expects JSON: {"text": "..."}.
    Returns JSON: {"response": "..."} — see process_answer() for what this
    might contain (feedback + next question, feedback + a follow-up, or the
    closing message).
    """
    data = request.get_json(silent=True) or {}
    answer_text = (data.get("text") or "").strip()
    if not answer_text:
        return jsonify({"error": "Kein Text eingegeben."}), 400

    pieces = process_answer(answer_text)
    response = " ".join(p["text"] for p in pieces).strip()
    print(f"[interview] final response sent to browser: {response!r}")
    return jsonify({"response": response})


@app.route("/api/chat/audio", methods=["POST"])
def chat_audio_endpoint():
    """
    Voice-condition endpoint: the participant spoke into the browser's microphone.
    Expects multipart/form-data with an "audio" file and optional "language" field.

    Streams the reply back as NDJSON (one JSON object per line), instead of
    waiting for the whole reply to be synthesized before responding — this
    is what Step 5b of the project plan is for: the browser can start
    playing the first sentence while later ones are still being
    synthesized. Line types:

      {"type": "transcript", "transcript": "..."}
        Sent first, as soon as speech-to-text is done.
      {"type": "sentence", "text": "...", "audio": "...base64 wav...", "kind": "...", "is_last": bool}
        One per sentence, in speaking order. "kind" is "feedback",
        "question", or "closing" — see process_answer() — and is what lets
        the browser's "Frage wiederholen" replay only the question later.
      {"type": "error", "message": "..."}
        Sent instead of a "sentence" line if synthesizing one sentence
        fails; playback of anything already sent is unaffected.
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
    pieces = process_answer(transcript)
    print(f"[interview] final response sent to browser: {' '.join(p['text'] for p in pieces)!r}")

    def generate():
        turn_start = time.time()

        def elapsed():
            return f"{time.time() - turn_start:.2f}s"

        print(f"[stream] transcript sent at +{elapsed()}")
        yield json.dumps({"type": "transcript", "transcript": transcript}) + "\n"

        # Short pause appended after every sentence except the last, so
        # consecutive audio pieces played back-to-back in the browser still
        # sound like natural speech instead of running into each other —
        # same reasoning as the pause between sentences in backend/tts.py.
        gap_seconds = 0.3

        for i, piece in enumerate(pieces):
            is_last = (i == len(pieces) - 1)
            sentence_text = piece["text"]
            try:
                sample_rate, audio_array = tts.synthesize(sentence_text)
                if not is_last:
                    gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
                    audio_array = np.concatenate([audio_array, gap])
                audio_base64 = speech.encode_wav_base64(sample_rate, audio_array)
                print(f"[stream] sentence {i + 1}/{len(pieces)} ({piece['kind']}) sent at +{elapsed()}: {sentence_text!r}")
                yield json.dumps({
                    "type": "sentence",
                    "text": sentence_text,
                    "audio": audio_base64,
                    "kind": piece["kind"],
                    "is_last": is_last,
                }) + "\n"
            except Exception as e:
                print(f"[stream] error at +{elapsed()}: {e}")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                break

    return Response(generate(), mimetype="application/x-ndjson")


# --- Terminal mode (used without --web) -------------------------------------
def run_terminal_mode():
    """
    A simple command-line loop: press Enter to record, speak, press Enter
    again to stop, and the assistant replies out loud. Useful for quick local
    testing without opening a browser.
    """
    print("\U0001F916 Local Voice Assistant")
    print("\u2500" * 42)
    selected_language = input("Which language should I use for transcription? (e.g. de, en, fr): ").strip().lower() or "en"

    print(f"TTS voice: {_tts_voice}")
    print(f"LLM model: {args.model or ('gemma3' if args.provider == 'ollama' else 'MiniMax-M2.7')}")
    print(f"LLM provider: {args.provider}")
    print("\u2500" * 42)
    print("Press Ctrl+C to exit.\n")

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0
    interview.reset()

    def speak(text: str):
        """
        Synthesize and play text out loud, sentence by sentence — the
        first sentence starts playing as soon as it's ready, while later
        sentences are still being synthesized, instead of waiting for the
        whole reply (see the project plan, Step 5 "reduce latency").

        Honors --save-voice by collecting the sentences actually played
        and writing them to disk afterwards as one file, instead of
        synthesizing everything a second time just for saving.
        """
        nonlocal response_count
        sample_rate = None
        played_chunks = []

        for sr, audio_array in tts.synthesize_streaming(text):
            sample_rate = sr
            speech.play_audio(sr, audio_array)
            if args.save_voice:
                played_chunks.append(audio_array)

        if args.save_voice and played_chunks:
            response_count += 1
            filename = f"voices/response_{response_count:03d}.wav"
            full_audio = np.concatenate(played_chunks)
            speech.write_wav_file(filename, sample_rate, full_audio)
            print(f"Voice saved to: {filename}")

    try:
        # The framing message and the first fixed question open the
        # interview — presented verbatim, no LLM call needed.
        first_question = interview.current_question()
        if first_question is None:
            print(f"Assistant: {interview.framing_message}")
            speak(interview.framing_message)
            print("(This interview guide has no questions yet.)")
            return
        opening = f"{interview.framing_message} {first_question['text']}".strip()
        print(f"Assistant: {opening}")
        speak(opening)

        while not interview.is_finished():
            input("\U0001F3A4 Press Enter to start recording your answer, then press Enter again to stop.")

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
            answer_text = stt.transcribe(audio_np, language=selected_language)
            print(f"You: {answer_text}")

            print("Generating response...")
            pieces = process_answer(answer_text)
            combined = " ".join(p["text"] for p in pieces).strip()

            print(f"Assistant: {combined}")
            speak(combined)

    except KeyboardInterrupt:
        print("\nExiting...")

    print("Session ended. Thank you for using the assistant!")


if __name__ == "__main__":
    if args.web:
        print("\U0001F916 Starting Flask web server for Local Voice Assistant")
        print("Open http://127.0.0.1:5000 in your browser to connect.")
        # threaded=True so the streamed /api/chat/audio response (Step 5b)
        # is served promptly rather than potentially waiting behind other
        # requests (e.g. the browser's static-file polling).
        app.run(host="127.0.0.1", port=5000, threaded=True)
    else:
        run_terminal_mode()
