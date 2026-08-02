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
import random
import re
import threading
import time
from queue import Queue

import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

import llm_provider
import speech
import stt
import study_settings
from behavior_profiles import load_behavior_profile
from interview_guide import InterviewGuide
from tts import TextToSpeechService

# Loads variables from a local .env file (if present) into the environment,
# e.g. MINIMAX_API_KEY, so it doesn't need to be exported in the shell every
# time. .env is gitignored — never commit it.
load_dotenv()

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

# --- Safety-concern handling (see e.g. study_content/prompts/nba_conversational.txt) ---
# Some behavior profiles instruct the model to reply with this exact marker,
# instead of normal FEEDBACK/FOLLOWUP text, when the participant's answer
# suggests acute psychological distress or a safety risk. It must never
# reach the participant as text or speech, and must pause the interview
# (no further advancing) instead — see the check right after every
# get_feedback_and_followup() call in process_answer() below.
SAFETY_CONCERN_MARKER = "[SAFETY_CONCERN]"
SAFETY_ABORT_MESSAGE = {
    "de": "Vielen Dank für Ihre Antworten. Wir beenden das Interview an dieser Stelle.",
    "en": "Thank you for your answers. We are ending the interview at this point.",
}
# Module-level flag: once tripped, every further turn just repeats the
# abort message below and never touches the interview guide's state again
# (mirrors the "single fixed session, one participant at a time" design
# already used for chat_sessions in llm_provider.py).
_safety_concern_triggered = False


def _safety_abort_message() -> str:
    return SAFETY_ABORT_MESSAGE.get(args.language, SAFETY_ABORT_MESSAGE["de"])


def _raise_safety_concern_alert() -> None:
    """
    Called the moment the model reports SAFETY_CONCERN_MARKER. Prints a
    hard-to-miss console alert — the console is where every other
    "[interview] ..." status line already goes (see process_answer()), and
    the study lead is present at every session per the ethics approval.
    """
    global _safety_concern_triggered
    _safety_concern_triggered = True
    print("=" * 60)
    print("[SAFETY ALERT] The model flagged a possible safety concern.")
    print("The interview has been paused. Please check in with the participant.")
    print("=" * 60)


# --- FEEDBACK style rotation (computed here, not left to the model) -------
# See study_content/prompts/nba_conversational.txt's "FEEDBACK styles"
# section: style 1 (Spiegeln) / 2 (Kleine echte Reaktion) / 3 (Konkretes
# Aufgreifen) rotation is driven entirely by this counter via
# llm_provider.format_required_style_context(), not by asking the model to
# track "never twice in a row" itself — a known reliability problem for
# smaller local models over many turns; the earlier prompt-only version of
# this rule was observed to be ignored, with the model settling into a
# single repeated template (e.g. always "Klingt nach ...") regardless of
# what was actually said.
#
# Only increments for genuine answering turns — a turn that turns into a
# clarification request (REPEAT_QUESTION) doesn't count, so the same style
# is simply tried again once the participant actually answers.
_style_turn_count = 0


def _next_required_style_context() -> str:
    style_index = (_style_turn_count % 3) + 1
    return llm_provider.format_required_style_context(style_index)


# --- Clarification-question handling (see e.g. nba_conversational.txt) ----
# Some behavior profiles let the model recognize that the participant asked
# a comprehension question about the current item instead of answering it,
# and signal that with this marker instead of a real follow-up. On match,
# the item is re-presented unchanged (see process_answer() below) instead
# of being scored against its aspects checklist or advanced past.
REPEAT_QUESTION_MARKER = "[REPEAT_QUESTION]"

# How many times the SAME item (fixed question, and/or its one follow-up —
# see InterviewGuide.register_clarification_request()) may be re-presented
# after a clarification question before giving up automated handling and
# handing off to the study lead, who is present throughout the session per
# the ethics approval. Prevents an unanswerable question from looping
# forever.
MAX_CLARIFICATIONS_PER_ITEM = 2

CLARIFICATION_HANDOFF_MESSAGE = {
    "de": "Diese Frage klären wir am besten gemeinsam mit der Studienleitung — sie hilft Ihnen gleich persönlich weiter.",
    "en": "Let's clarify this question together with the study lead instead — they'll help you with it in person in a moment.",
}


def _clarification_handoff_message() -> str:
    return CLARIFICATION_HANDOFF_MESSAGE.get(args.language, CLARIFICATION_HANDOFF_MESSAGE["de"])


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


def _response_pieces(feedback: str, next_question: dict | None) -> list[str]:
    """
    Break the reply that follows a piece of feedback into the individual
    sentences it's made of, in speaking order — this is what lets the voice
    condition start speaking the first sentence while later ones are still
    being synthesized (see Step 5b of the project plan).

    For "natural_transition" delivery, the model blends the feedback and
    the next question into one smoothly-worded message — there are no
    separately-known pieces in that case, only sentences within the result.

    For "verbatim" delivery with profile.transition_phrases set (see
    study_content/prompts/profiles.yaml), a randomly-picked lead-in phrase
    is prepended before the next question — only when there IS a next
    question (never before the closing message) — so feedback and question
    don't just get stitched together with no transition. The question's
    own wording is never touched, only bridged into.
    """
    following_text = next_question["text"] if next_question is not None else interview.closing_message

    if profile.question_delivery == "natural_transition":
        combined = llm_provider.get_natural_transition(llm, feedback, following_text, language_code=args.language)
        return _split_sentences(combined)

    if next_question is not None and profile.transition_phrases:
        following_text = f"{random.choice(profile.transition_phrases)} {following_text}"

    return _split_sentences(feedback) + _split_sentences(following_text)


def _handle_clarification_request(feedback: str, repeated_text: str) -> list[str]:
    """
    Called when the model reports REPEAT_QUESTION_MARKER instead of a real
    follow-up: the participant asked a comprehension question rather than
    answering. Speaks the model's FEEDBACK (the clarification) as usual,
    then either re-presents the same item unchanged (repeated_text — the
    fixed question or the pending follow-up, whichever is currently
    active) or, once MAX_CLARIFICATIONS_PER_ITEM is exceeded, hands off to
    the study lead and moves the guide on instead of looping forever.

    Does NOT call interview.start_followup() or otherwise touch follow-up
    state — the caller's own state (awaiting-followup or not) is left
    exactly as it was, so the next answer is still matched against the
    same item.
    """
    count = interview.register_clarification_request()
    print(f"[interview] clarification requested (count={count}/{MAX_CLARIFICATIONS_PER_ITEM}) — re-presenting current item")
    if count > MAX_CLARIFICATIONS_PER_ITEM:
        next_question = interview.advance()
        return _response_pieces(_clarification_handoff_message(), next_question)
    return _split_sentences(feedback) + _split_sentences(repeated_text)


# Backstop for when the model recognizes a clarification request in its
# FEEDBACK wording (e.g. explains what "Pause" means) but doesn't actually
# use REPEAT_QUESTION_MARKER in FOLLOWUP as instructed — observed to happen
# with this local model. Matched against the PARTICIPANT's own text, not
# the model's response, so it fires regardless of what the model did.
_CLARIFICATION_PATTERNS = [
    r'was meinen sie',
    r'wie meinen sie',
    r'was verstehen sie unter',
    r'was hei[ßs]t das',
    r'was ist damit gemeint',
    r'k[oö]nnen sie das (genauer )?erkl[aä]ren',
    r'das (habe ich|verstehe ich)(?: jetzt| gerade)? nicht(?: verstanden)?',
]


def _looks_like_clarification_request(answer_text: str) -> bool:
    normalized = answer_text.lower()
    return any(re.search(pattern, normalized) for pattern in _CLARIFICATION_PATTERNS)


def process_answer(answer_text: str) -> list[str]:
    """
    The core interview turn: given what the participant just said, decide
    what the assistant says next, and advance the interview guide's state
    accordingly. Shared by both web routes and the terminal mode, so the
    behavior is identical regardless of modality.

    Returns a list of individual sentences, in speaking order, rather than
    one combined string — this is what backend/app.py's audio endpoint
    streams to the browser one at a time (see Step 5b of the project plan).
    Callers that just want the full text (text condition, terminal mode)
    can simply join the list with spaces.

    Two cases:
      1. The participant is answering a follow-up that was asked last turn
         (interview.is_awaiting_followup_answer()) — no further follow-up
         is allowed here (capped at one per question), then the guide moves on.
      2. The participant is answering the current fixed question — a
         follow-up may be offered if this item allows it and none has been
         used for it yet.

    A third, terminal case: if the model ever responds with
    SAFETY_CONCERN_MARKER (see some behavior profiles' "Safety" section,
    e.g. study_content/prompts/nba_conversational.txt), the interview is
    paused for good — every call from that point on short-circuits at the
    top of this function and just repeats _safety_abort_message().

    A fourth case, in either of the two above: if the model responds with
    REPEAT_QUESTION_MARKER instead of a real follow-up (the participant
    asked a comprehension question about the item instead of answering
    it), see _handle_clarification_request() — the same item is
    re-presented unchanged, up to MAX_CLARIFICATIONS_PER_ITEM times, before
    handing off to the study lead instead of looping forever. This also
    fires if _looks_like_clarification_request(answer_text) matches, even
    when the model didn't use the marker — this local model sometimes
    writes a genuine clarification into FEEDBACK but forgets to set
    REPEAT_QUESTION_MARKER, so the participant's own wording is checked too.

    Prints a short debug line to the console for every turn (visible only
    in the terminal running this app, never sent to the participant) — this
    is meant to make it easy to check whether the follow-up mechanism is
    actually engaging as expected while testing, instead of having to guess
    from the final combined text alone.
    """
    global _style_turn_count

    if _safety_concern_triggered:
        return _split_sentences(_safety_abort_message())

    if interview.is_awaiting_followup_answer():
        print(f"[interview] answering follow-up: \"{interview.pending_followup_text}\"")
        feedback, followup = llm_provider.get_feedback_and_followup(
            chain_with_history, profile, interview.pending_followup_text, answer_text,
            followup_allowed=False, language_code=args.language,
            aspects_context=llm_provider.format_aspects_context(interview.current_question()),
            required_style_context=_next_required_style_context(),
        )
        if feedback.strip() == SAFETY_CONCERN_MARKER:
            _raise_safety_concern_alert()
            return _split_sentences(_safety_abort_message())
        if (followup and followup.strip() == REPEAT_QUESTION_MARKER) or _looks_like_clarification_request(answer_text):
            # Repeating the SAME generic maintenance_question after
            # explaining it doesn't help — escalate to a specific followup
            # from the same aspect instead, and keep pending_followup_text
            # in sync so the next turn is evaluated against what was
            # actually just re-presented (see escalate_to_specific_followup()).
            escalated = llm_provider.escalate_to_specific_followup(interview.current_question(), interview.pending_followup_text)
            if escalated != interview.pending_followup_text:
                print(f"[interview] escalating generic followup {interview.pending_followup_text!r} -> specific {escalated!r} after clarification")
                interview.pending_followup_text = escalated
            return _handle_clarification_request(feedback, interview.pending_followup_text)
        _style_turn_count += 1
        print(f"[interview] feedback={feedback!r}")
        next_question = interview.advance()
        return _response_pieces(feedback, next_question)

    question = interview.current_question()
    if question is None:
        # The guide was already finished; nothing left to ask.
        return _split_sentences(interview.closing_message)

    followup_allowed = question["allow_followup"] and not interview.followup_already_used()
    print(
        f"[interview] question={question['id']!r} "
        f"allow_followup(guide)={question['allow_followup']} "
        f"followup_allowed(this turn)={followup_allowed}"
    )
    feedback, followup = llm_provider.get_feedback_and_followup(
        chain_with_history, profile, question["text"], answer_text,
        followup_allowed=followup_allowed, language_code=args.language,
        aspects_context=llm_provider.format_aspects_context(question),
        required_style_context=_next_required_style_context(),
    )
    if feedback.strip() == SAFETY_CONCERN_MARKER:
        _raise_safety_concern_alert()
        return _split_sentences(_safety_abort_message())
    if (followup and followup.strip() == REPEAT_QUESTION_MARKER) or _looks_like_clarification_request(answer_text):
        return _handle_clarification_request(feedback, question["text"])
    _style_turn_count += 1

    if followup:
        # Never let the model's own invented wording reach the participant
        # for a checklist-based question — only its own maintenance_questions/
        # followups (or a safe generic fallback) are ever allowed through.
        # No-op for guides without "aspects" (see constrain_followup_to_checklist()).
        constrained_followup = llm_provider.constrain_followup_to_checklist(question, followup)
        if constrained_followup != followup:
            print(f"[interview] followup {followup!r} not in checklist — using {constrained_followup!r} instead")
        followup = constrained_followup

    print(f"[interview] feedback={feedback!r} followup={followup!r}")

    if followup:
        interview.start_followup(followup)
        return _split_sentences(feedback) + _split_sentences(followup)

    next_question = interview.advance()
    return _response_pieces(feedback, next_question)


def _opening_message() -> str:
    """
    Resets the interview guide to its start and returns the framing message
    plus the first fixed question, verbatim — shared by both
    /api/interview/start (text) and /api/interview/start/audio (voice).
    """
    global _style_turn_count
    interview.reset()
    _style_turn_count = 0
    first_question = interview.current_question()
    if first_question is None:
        # An empty guide (e.g. a guide file with no questions yet).
        return interview.framing_message
    return f"{interview.framing_message} {first_question['text']}".strip()


@app.route("/api/interview/start", methods=["GET"])
def interview_start_endpoint():
    """
    Called once when the participant opens the page: returns the framing
    message and the first fixed question, verbatim from the active
    interview guide. No LLM call is involved — there is nothing to
    generate yet, only fixed text to present.
    Returns JSON: {"message": "..."}.
    """
    return jsonify({"message": _opening_message()})


@app.route("/api/interview/start/audio", methods=["GET"])
def interview_start_audio_endpoint():
    """
    Voice-condition companion to /api/interview/start: streams the framing
    message + first question as synthesized audio, one NDJSON "sentence"
    line at a time — same wire format as /api/chat/audio (see
    _stream_synthesized_sentences() below), so the frontend's existing
    readStreamedReply() can play it exactly like any other reply, instead
    of only revealing the opening message as silent text.

    Called once, right when the participant clicks "Interview starten" (see
    playOpeningSpeech() in frontend/conversation.js) — not on page load,
    since audio should only ever play after a user gesture.
    """
    message = _opening_message()
    pieces = _split_sentences(message)
    return Response(_stream_synthesized_sentences(pieces), mimetype="application/x-ndjson")


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
    response = " ".join(pieces).strip()
    print(f"[interview] final response sent to browser: {response!r}")
    return jsonify({"response": response})


def _stream_synthesized_sentences(pieces: list[str]):
    """
    Shared NDJSON sentence-audio generator, used by both /api/chat/audio and
    /api/interview/start/audio. Synthesizes and yields one sentence at a
    time instead of waiting for the whole reply — this is what Step 5b of
    the project plan is for: the browser can start playing the first
    sentence while later ones are still being synthesized. Line types:

      {"type": "sentence", "text": "...", "audio": "...base64 wav...", "is_last": bool}
        One per sentence, in speaking order.
      {"type": "error", "message": "..."}
        Sent instead of a "sentence" line if synthesizing one sentence
        fails; playback of anything already sent is unaffected.
    """
    # Short pause appended after every sentence except the last, so
    # consecutive audio pieces played back-to-back in the browser still
    # sound like natural speech instead of running into each other — same
    # reasoning as the pause between sentences in backend/tts.py.
    gap_seconds = 0.3

    for i, sentence_text in enumerate(pieces):
        is_last = (i == len(pieces) - 1)
        try:
            sample_rate, audio_array = tts.synthesize(sentence_text)
            if not is_last:
                gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
                audio_array = np.concatenate([audio_array, gap])
            audio_base64 = speech.encode_wav_base64(sample_rate, audio_array)
            yield json.dumps({
                "type": "sentence",
                "text": sentence_text,
                "audio": audio_base64,
                "is_last": is_last,
            }) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            break


@app.route("/api/chat/audio", methods=["POST"])
def chat_audio_endpoint():
    """
    Voice-condition endpoint: the participant spoke into the browser's microphone.
    Expects multipart/form-data with an "audio" file and optional "language" field.

    Streams the reply back as NDJSON (one JSON object per line) — see
    _stream_synthesized_sentences() above for the "sentence"/"error" lines.
    Additionally sends, first:

      {"type": "transcript", "transcript": "..."}
        Sent first, as soon as speech-to-text is done.
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
    print(f"[interview] final response sent to browser: {' '.join(pieces)!r}")

    def generate():
        yield json.dumps({"type": "transcript", "transcript": transcript}) + "\n"
        yield from _stream_synthesized_sentences(pieces)

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
            combined = " ".join(pieces).strip()

            print(f"Assistant: {combined}")
            speak(combined)

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
