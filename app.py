import time
import threading
import re
import io
import base64
import wave
import numpy as np
import whisper
import sounddevice as sd
from scipy.signal import resample_poly
import argparse
import os
from queue import Queue
from rich.console import Console
from flask import Flask, request, jsonify
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("base")  # Load multilingual Whisper model for speech-to-text

# Parse command line arguments
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
selected_language = args.language

# Initialize TTS with ChatterBox
tts = TextToSpeechService()
app = Flask(__name__, static_folder="web_interface_new", static_url_path="")

@app.route("/")
def index():
    """
    Serve the web interface HTML page.
    """
    return app.send_static_file("index.html")

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    API endpoint for browser-based chat.
    Expects JSON: {"text": "...", "language": "de"}.
    Returns JSON: {"response": "..."}.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language = data.get("language") or args.language
    if not text:
        return jsonify({"error": "Kein Text eingegeben."}), 400

    response = get_llm_response(text)
    return jsonify({"response": response})


@app.route("/api/chat/audio", methods=["POST"])
def chat_audio_endpoint():
    """
    API endpoint for browser-based speech chat.
    Expects multipart/form-data with an "audio" file and optional "language" field.
    Returns JSON: {"transcript": "...", "response": "...", "audio": "...base64..."}.
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file uploaded."}), 400

    language = request.form.get("language") or args.language
    try:
        audio_bytes = audio_file.read()
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio_np = audio_np.reshape(-1, channels).mean(axis=1)
        if sr != 16000:
            audio_np = resample_poly(audio_np, 16000, sr)
    except Exception as e:
        return jsonify({"error": f"Unable to decode audio: {e}"}), 400

    transcript = transcribe(audio_np, language=language)
    response = get_llm_response(transcript)
    sample_rate, audio_array = tts.long_form_synthesize(
        response,
        audio_prompt_path=args.voice,
        exaggeration=analyze_emotion(response),
        cfg_weight=args.cfg_weight,
    )

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        clipped = np.clip(audio_array, -1.0, 1.0)
        int16_audio = (clipped * 32767).astype(np.int16)
        wf.writeframes(int16_audio.tobytes())
    audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")

    return jsonify({
        "transcript": transcript,
        "response": response,
        "audio": audio_base64,
    })


def create_llm(provider: str, model: str | None = None, api_key: str | None = None, temperature: float = 0.7):
    """
    Create an LLM instance based on the selected provider.

    Args:
        provider: LLM provider name ('ollama' or 'minimax').
        model: Model name. Defaults to 'gemma3' for ollama, 'MiniMax-M2.7' for minimax.
        api_key: API key for cloud providers (or set MINIMAX_API_KEY env var).
        temperature: LLM temperature (default: 0.7).

    Returns:
        A LangChain LLM or ChatModel instance.
    """
    if provider == "ollama":
        return OllamaLLM(model=model or "gemma3", base_url="http://localhost:11434")
    elif provider == "minimax":
        from langchain_openai import ChatOpenAI

        resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not resolved_key:
            raise ValueError(
                "MiniMax API key is required. Set the MINIMAX_API_KEY environment "
                "variable or pass --api-key on the command line."
            )
        # MiniMax temperature must be in (0.0, 1.0]
        clamped_temperature = max(0.01, min(1.0, temperature))
        return ChatOpenAI(
            model=model or "MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            api_key=resolved_key,
            temperature=clamped_temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, minimax")


# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. Always answer in the same language as the user's message. Keep replies polite, concise, and under 20 words."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM via provider factory
llm = create_llm(
    provider=args.provider,
    model=args.model,
    api_key=args.api_key,
    temperature=args.temperature,
)

# Create the chain with modern LCEL syntax
# StrOutputParser normalizes output across providers (string from Ollama, AIMessage from ChatOpenAI)
chain = prompt_template | llm | StrOutputParser()

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray, language: str = "de") -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.
        language (str): The language code for transcription.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False, language=language)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Use a default session ID for this simple voice assistant
    session_id = "voice_assistant_session"

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    # Normalize common assistant prefixes that some models return.
    # This prevents repeated labels like "AI: AI:" from appearing in the chat.
    return normalize_model_response(response)


def normalize_model_response(text: str) -> str:
    """
    Remove common assistant labels from model output.

    Some chat models may return answers prefixed with labels like
    "AI:" or "Assistant:". We strip those prefixes to avoid repeated
    labels when the response is stored in history.
    """
    normalized = (text or "").strip()
    while True:
        new_text = re.sub(r'^(?:AI:|Assistant:)\s*', '', normalized, flags=re.IGNORECASE)
        if new_text == normalized:
            break
        normalized = new_text.strip()
    return normalized.strip()


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


if __name__ == "__main__":
    # If the program is started with --web, run it as a Flask web server.
    if args.web:
        console.print("[cyan]🤖 Starting Flask web server for Local Voice Assistant")
        console.print("[cyan]Open http://127.0.0.1:5000 in your browser to connect.")
        # Flask starts the server here. The web UI must be opened separately.
        app.run(host="127.0.0.1", port=5000)
    else:
        # Standard-Modus: Terminal-basierte Aufnahme und Wiedergabe.
        console.print("[cyan]🤖 Local Voice Assistant with ChatterBox TTS")
        console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        selected_language = input("Which language should I use for transcription? (e.g. de, en, fr): ").strip().lower() or "en"

        if args.voice:
            console.print(f"[green]Using voice cloning from: {args.voice}")
        else:
            console.print("[yellow]Using default voice (no cloning)")

        console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
        console.print(f"[blue]CFG weight: {args.cfg_weight}")
        console.print(f"[blue]LLM model: {args.model or ('gemma3' if args.provider == 'ollama' else 'MiniMax-M2.7')}")
        console.print(f"[blue]LLM provider: {args.provider}")
        console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        console.print("[cyan]Press Ctrl+C to exit.\n")

        # Create voices directory if saving voices
        if args.save_voice:
            os.makedirs("voices", exist_ok=True)

        response_count = 0

        try:
            while True:
                console.input(
                    "🎤 Press Enter to start recording, then press Enter again to stop."
                )

                data_queue = Queue()  # type: ignore[var-annotated]
                stop_event = threading.Event()
                recording_thread = threading.Thread(
                    target=record_audio,
                    args=(stop_event, data_queue),
                )
                recording_thread.start()

                input()
                stop_event.set()
                recording_thread.join()

                audio_data = b"".join(list(data_queue.queue))
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if audio_np.size > 0:
                    with console.status("Transcribing...", spinner="dots"):
                        text = transcribe(audio_np)
                    console.print(f"[yellow]You: {text}")

                    with console.status("Generating response...", spinner="dots"):
                        response = get_llm_response(text)

                        # Analyze emotion and adjust exaggeration dynamically
                        dynamic_exaggeration = analyze_emotion(response)

                        # Use lower cfg_weight for more expressive responses
                        dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                        sample_rate, audio_array = tts.long_form_synthesize(
                            response,
                            audio_prompt_path=args.voice,
                            exaggeration=dynamic_exaggeration,
                            cfg_weight=dynamic_cfg
                        )

                    console.print(f"[cyan]Assistant: {response}")
                    console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                    # Save voice sample if requested
                    if args.save_voice:
                        response_count += 1
                        filename = f"voices/response_{response_count:03d}.wav"
                        tts.save_voice_sample(response, filename, args.voice)
                        console.print(f"[dim]Voice saved to: {filename}[/dim]")

                    play_audio(sample_rate, audio_array)
                else:
                    console.print(
                        "[red]No audio recorded. Please ensure your microphone is working."
                    )

        except KeyboardInterrupt:
            console.print("\n[red]Exiting...")

        console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
