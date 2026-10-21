# Legacy API Baseline (Phase 0.4)

> **Purpose:** Snapshot of the *current* (legacy) HTTP API of
> `backend/app.py`, written as the reference for the A/B behavior comparison
> in Phase 8.1 of `RECONSTRUCTION_PLAN.md` (`--pipeline legacy|new`).
> If the legacy pipeline changes before cutover, this document must be
> updated, or the A/B comparison loses its baseline.
>
> Snapshot: 2026-10-19 · branch `Z.-Version` · source of truth:
> `backend/app.py` (line references below are to this file)

## 1. Server & global state

- Flask app bound to `127.0.0.1:5000`, run with `threaded=True`
  (`app.py:533`), started only with `--web`.
- **Global mutable state** (deliberate "one participant at a time" model):
  - one `InterviewGuide` instance, loaded at startup from the guide named in
    `study_content/study_settings.yaml` (`interview_guide:
    depression_scid_example.yaml`), `app.py:78-83`;
  - one LLM chain with in-memory chat history and the **fixed**
    `session_id = "voice_assistant_session"` (`llm_provider.py`);
  - one Piper TTS voice (`study_settings.yaml` → `tts_voice`, default
    `de_DE-thorsten-high`), one faster-whisper STT model.
- **No persistence.** Everything is in memory; only `--save-voice` writes to
  disk (WAV files under `voices/`).
- `interview.reset()` is called by **both** `start` endpoints — a new
  participant (or a page reload) restarts the guide from the first question.
  `reset()` does **not** clear the LLM's in-memory chat history.
- Terminal mode (without `--web`) shares `process_answer()`, so text and
  voice/terminal behavior are identical by construction.
- CLI knobs that influence API behavior: `--provider ollama|minimax`
  (default `ollama`), `--model`, `--api-key`/`MINIMAX_API_KEY`,
  `--temperature` (default 0.7), `--language` (default from
  `study_settings.yaml`, else `de`), `--save-voice`.

## 2. The four API endpoints

### 2.1 `GET /api/interview/start` — opening, text condition (`app.py:246-262`)

- **Request:** no body, no parameters.
- **Behavior:** resets the guide; returns framing message + first fixed
  question **verbatim** from the guide YAML. **No LLM call.**
- **Response 200:**
  ```json
  {"message": "Guten Tag und vielen Dank, dass Sie sich Zeit nehmen. Ich bin ein KI-gestütztes System und werde Ihnen im Rahmen dieser Studie einige Fragen zu Ihrem Befinden in den letzten zwei Wochen stellen. Es gibt keine richtigen oder falschen Antworten — antworten Sie einfach so, wie es für Sie zutrifft. Wie würden Sie Ihre Stimmung in den letzten zwei Wochen beschreiben?"}
  ```
  (framing and first question joined with a single space; if the guide has no
  questions, `message` is the framing message alone)
- **Errors:** none defined (any exception → Flask default 500).
- **Side effects:** guide state reset; LLM not touched.

### 2.2 `GET /api/interview/start/audio` — opening, voice condition (`app.py:265-298`)

- **Request:** no body, no parameters.
- **Behavior:** same reset as 2.1; then synthesizes framing + first question
  **sentence by sentence** with Piper and returns everything in **one** JSON
  object (deliberately *not* streamed — the frontend fetches this in the
  background while the start overlay is still showing, so audio is ready
  before "Interview starten" is clicked).
- **Response 200:**
  ```json
  {
    "pieces": [
      {"text": "<sentence>", "audio": "<base64 WAV>", "kind": "framing"},
      {"text": "<sentence>", "audio": "<base64 WAV>", "kind": "question"}
    ]
  }
  ```
  - `kind` is `"framing"` for the greeting, `"question"` for the first fixed
    question — same tagging scheme as later turns, so "Frage wiederholen"
    (repeat question) replays only the question.
  - `audio` is a base64-encoded WAV (Piper sample rate, 16-bit PCM).
  - Sentences are split with the punctuation regex `(?<=[.!?])\s+`
    (`_split_sentences`, `app.py:124-143`).
- **Errors:** none defined (synthesis exception → Flask default 500).
- **Side effects:** guide state reset; TTS synthesis for the whole opening.

### 2.3 `POST /api/chat` — answer turn, text condition (`app.py:301-329`)

- **Request:** JSON body `{"text": "<participant answer>"}`.
- **Behavior:**
  1. `safety_check.check_for_safety_concern(answer, language)` — keyword
     check on the participant's own input against
     `study_content/safety_keywords.yaml`. Result is a plain boolean,
     **non-blocking**: the interview continues regardless.
  2. `process_answer(answer)` — see §3.
- **Response 200:**
  ```json
  {
    "response": "<feedback> <next question | follow-up | closing message>",
    "finished": false,
    "safety_flag": false
  }
  ```
  - `response`: all returned pieces joined with single spaces (feedback +
    next fixed question, or feedback + follow-up, or just the closing
    message when the guide is done).
  - `finished`: `interview.is_finished()` — `true` on the turn that delivers
    the closing message, and on every later turn (the closing message is
    then returned again). The frontend uses it to stop accepting answers.
  - `safety_flag`: `true` iff the participant's answer matched a safety
     keyword; frontend shows the calm, non-blocking hint with support
     resources.
- **Error 400:** `{"error": "Kein Text eingegeben."}` when `text` is empty or
  missing (note: German error string, unlike the English ones in 2.4).
- **LLM calls per turn:** 1 (`get_feedback_and_followup`); **plus 1 more** if
  the active behavior profile has `question_delivery: natural_transition`
  (the `flexible_conversational` profile — `get_natural_transition` blends
  feedback and next question into one message).
- **Side effects:** safety check logged to console; guide state advanced or
  follow-up started; LLM history grows (in memory, fixed session id).

### 2.4 `POST /api/chat/audio` — answer turn, voice condition (`app.py:332-420`)

- **Request:** `multipart/form-data` with file field `audio` (WAV) and
  optional form field `language` (falls back to the `--language` value).
- **Behavior** (all synchronous, before the stream starts):
  1. Decode WAV → float32 mono, resampled to 16 kHz
     (`speech.decode_uploaded_wav`).
  2. Transcribe with faster-whisper (`stt.transcribe`, language as above).
  3. Safety check on the **transcript** (same semantics as 2.3).
  4. `process_answer(transcript)`; `finished` computed once, up front.
  5. Stream the reply sentence by sentence: each sentence is TTS-synthesized
     when it is the stream's turn (first sentence is playable while later
     ones are still being synthesized).
- **Response 200:** NDJSON stream, `Content-Type: application/x-ndjson`, one
  JSON object per line, in this order:
  1. First line, as soon as STT is done:
     ```json
     {"type": "transcript", "transcript": "<recognized answer>", "safety_flag": false}
     ```
  2. One line per sentence, in speaking order:
     ```json
     {"type": "sentence", "text": "<sentence>", "audio": "<base64 WAV>", "kind": "feedback", "is_last": false, "finished": false}
     ```
     - `kind` is `"feedback"`, `"question"`, or `"closing"` (see §3).
     - `is_last`: `true` on the final sentence of the reply.
     - `finished`: the up-front `is_finished()` value (same semantics as 2.3).
     - A 0.3 s silence gap is appended to every sentence's audio **except**
       the last, so back-to-back browser playback sounds natural.
  3. Instead of a `sentence` line, if one synthesis fails:
     ```json
     {"type": "error", "message": "<exception text>"}
     ```
     and the stream **stops** (sentences already sent keep playing).
- **Error 400:**
  - `{"error": "No audio file uploaded."}` — no `audio` field;
  - `{"error": "Unable to decode audio: <exception text>"}` — bad WAV.
- **Side effects:** as 2.3, plus an STT run and per-sentence TTS runs.

## 3. Shared turn pipeline — `process_answer()` (`app.py:182-243`)

Called by `/api/chat`, `/api/chat/audio` and the terminal mode; returns
`[{"text", "kind"}, ...]` in speaking order.

1. **Pending follow-up?** (`interview.is_awaiting_followup_answer()`):
   the answer is evaluated with `followup_allowed=False` (one follow-up per
   question, max), then `interview.advance()` → next fixed question or
   closing. Pieces: feedback (`kind=feedback`) + next question
   (`kind=question`) or closing (`kind=closing`) — or, for the
   `natural_transition` profile, one blended LLM-generated message, all
   tagged `question`.
2. **Current fixed question:** feedback + possibly a follow-up. A follow-up
   is offered only if the question has `allow_followup: true` and none has
   been used for it yet. If asked: `interview.start_followup(followup)`,
   pieces = feedback + follow-up (`kind=question`). Otherwise
   `interview.advance()` and pieces as in case 1.
3. **Guide already finished:** closing message only (`kind=closing`) — this
   also fires for any extra `/api/chat` call after the end (the closing
   message is repeated).

Sentence splitting is punctuation-based (`(?<=[.!?])\s+`), not
abbreviation-aware — good enough for short replies, and Piper does its own
splitting for the actual audio.

## 4. Auxiliary (non-API) routes

- `GET /` → serves `frontend/index.html` (`app.py:107-110`).
- `GET /study-assets/<path:filename>` → files from `study_content/assets/`
  (logo / embodiment images; deliberately the only reachable subfolder of
  `study_content/`), `app.py:113-121`.

## 5. Known legacy behaviors (what the A/B comparison in Phase 8 must diff)

1. **Response text sequence** per answer: strict profile ⇒
   `"<feedback> <verbatim next question>"`; flexible profile ⇒ blended
   `natural_transition` text; follow-up turns ⇒ `"<feedback> <follow-up>"`;
   last turn ⇒ closing message.
2. **`finished` semantics:** true on the closing turn and on all later turns
   (closing message re-delivered).
3. **`safety_flag` semantics:** single boolean from a keyword list on the
   participant's input only; non-blocking; no other safety channel.
4. **Guide reset on every `start` call** (including page reload); LLM chat
   history is *not* reset.
5. **Delivery model:** fire-and-forget — no reservation, no `delivery_id`,
   no ACK, no retransmission; a client disconnect mid-stream loses the rest
   of the reply.
6. **One participant at a time** (global state); concurrent requests would
   interleave on the same guide instance.
7. **Language drift is constrained** by an explicit language directive in
   the prompt (see `tests/test_llm_provider.py::TestLanguageEnforcement`).

*End of baseline. For the target-architecture counterpart of each item, see
`RECONSTRUCTION_PLAN.md` §3 (Phases 2–5).*
