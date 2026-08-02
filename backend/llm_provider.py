"""
LLM PROVIDER — TALKING TO THE LANGUAGE MODEL
==============================================
This file is responsible for everything related to generating the
assistant's text reply: which language model to use, remembering the
conversation so far, and cleaning up the model's raw output.

It does NOT know anything about audio, microphones, or speakers — it only
deals with text in, text out. This is on purpose: the same code here works
whether the participant is typing or talking, since by the time text reaches
this file, speech has already been converted to text (see backend/stt.py).

IMPORTANT FOR THE STUDY: This file no longer lets the model chat freely.
As of Step 2 of the project plan ("Interview guide from files"), the model
only ever generates a short acknowledgement of the participant's answer —
the actual questions always come verbatim from
study_content/interviews/*.yaml (see backend/interview_guide.py). As of
Step 3 ("Constrain the LLM"), the model's actual instructions come from a
researcher-editable behavior profile (see backend/behavior_profiles.py and
study_content/prompts/), and the model replies in a structured
"FEEDBACK: ... / FOLLOWUP: ..." format so the code can reliably tell
whether a follow-up question was asked (see parse_feedback_and_followup()
below).

Used by: backend/app.py (the Flask web routes) and the terminal mode.
"""
import difflib
import os
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM

# Maps a study_settings.yaml language code to a human-readable name that is
# inserted directly into the prompt (see get_feedback_and_followup() and
# get_natural_transition() below). This is deliberately explicit rather than
# relying on the model to infer the participant's language on its own —
# smaller local models are not reliable at that, and were observed drifting
# into English regardless of what the participant actually said.
LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
}


def _language_name(language_code: str) -> str:
    """Turn a study_settings.yaml language code into a name usable in a prompt."""
    return LANGUAGE_NAMES.get(language_code, language_code)


def create_llm(provider: str, model: str | None = None, api_key: str | None = None, temperature: float = 0.7):
    """
    Create a connection to the language model that will generate replies.

    Args:
        provider: Which LLM provider to use: "ollama" (a model running fully
            locally on this computer — the default, and the only option that
            keeps all study data on this machine) or "minimax" (a cloud API,
            currently only useful for quick testing, NOT for real study data).
        model: Model name. Defaults to "gemma3" for ollama, "MiniMax-M2.7" for minimax.
        api_key: API key for the cloud provider (or set the MINIMAX_API_KEY
            environment variable instead of passing it here).
        temperature: How much randomness/creativity the model uses in its
            replies (0.0 = very predictable, 1.0 = more varied). Default 0.7.

    Returns:
        A LangChain LLM object that backend/app.py and the terminal mode can
        send messages to.
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
        # MiniMax only accepts a temperature strictly between 0 and 1.
        clamped_temperature = max(0.01, min(1.0, temperature))
        return ChatOpenAI(
            model=model or "MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            api_key=resolved_key,
            temperature=clamped_temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, minimax")


# The feedback prompt. Its actual behavior instructions ({instructions}) come
# from a researcher-editable behavior profile (see backend/behavior_profiles.py)
# rather than being fixed here — this template only supplies the *shape* of
# the conversation, not the tone/strictness rules themselves.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{instructions}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", (
        "I was asked: \"{question}\"\n"
        "{aspects_context}"
        "{required_style_context}"
        "I answered: \"{answer}\"\n"
        "(Follow-up for this item: {followup_status}.)"
    )),
])

# A second, independent prompt used only for behavior profiles whose
# question_delivery is "natural_transition" (see profiles.yaml): it takes an
# already-generated feedback sentence and the next fixed question, and asks
# the model to smoothly reword the transition between them — without
# changing what is actually being asked. This is intentionally a separate,
# stateless call (no conversation history) since it is a wording task, not
# part of the interview itself.
transition_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You will be given a short acknowledgement sentence and the exact "
        "next question that must be asked. Rewrite them together as one "
        "smooth, natural-sounding message, written in {language_name}. You "
        "MUST preserve the acknowledgement's meaning and the question's "
        "exact meaning and everything it asks for — only adjust the "
        "phrasing/wording for a natural transition, never add, remove, or "
        "change what is being asked. Reply with only the combined message, "
        "nothing else."
    )),
    ("human", "Acknowledgement: \"{feedback}\"\nNext question (preserve meaning exactly): \"{next_question}\""),
])

# Keeps track of each participant's conversation history in memory, so the
# model "remembers" what was said earlier in the same session. Keyed by a
# session id (currently there is only ever one participant at a time, as
# decided in the project plan, so a single fixed session id is used).
chat_sessions: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get the conversation history for a session, creating it if it doesn't exist yet."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


def build_chain(llm):
    """
    Wire the prompt template, the model, and the conversation history together
    into one pipeline that can be called with a single line: chain.invoke(...).
    """
    chain = prompt_template | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="answer",
        history_messages_key="history",
    )


def _strip_leading_junk(text: str) -> str:
    """
    Remove a stray leading comma/whitespace some models produce (e.g.
    ", Das ist verständlich." instead of "Das ist verständlich."). Shared
    by normalize_model_response() (for the raw response as a whole) and
    parse_feedback_and_followup() (for the FEEDBACK/FOLLOWUP field
    contents individually — a leading comma there sits right after the
    "FEEDBACK:"/"FOLLOWUP:" label, not at the very start of the raw
    response, so normalize_model_response() alone never reached it).
    """
    return re.sub(r'^[,\s\u2028\u2029]+', '', text).strip()


def normalize_model_response(text: str) -> str:
    """
    Clean up the model's raw reply before showing it to the participant.

    Some models prefix their answer with a label like "AI:" or "Assistant:",
    or leave a stray leading comma. This strips that away so only the actual
    reply text remains.
    """
    normalized = (text or "").strip()
    while True:
        new_text = re.sub(r'^(?:AI:|Assistant:)\s*', '', normalized, flags=re.IGNORECASE)
        if new_text == normalized:
            break
        normalized = new_text.strip()

    return _strip_leading_junk(normalized)


def parse_feedback_and_followup(raw_response: str) -> tuple[str, str | None]:
    """
    Parse the model's structured "FEEDBACK: ...\\nFOLLOWUP: ..." response.

    This is the mechanism that lets the rest of the code reliably know
    whether the model asked a follow-up question, instead of guessing from
    free-form text. If the model doesn't follow the format at all (models
    occasionally deviate), the whole response is treated as the feedback
    and no follow-up is assumed — fails safe, rather than crashing.

    EXTENDING THIS: if you need something richer than a yes/no follow-up
    (e.g. checking whether an open-ended answer actually covered several
    required points, and asking more than one follow-up until it does),
    this function — together with get_feedback_and_followup() below and
    the follow-up state in backend/interview_guide.py — is the place to
    change. A plain-text line format like this one gets unwieldy once you
    need to return structured data (e.g. a list of covered/missing points);
    at that point, switching this function to parse JSON (or using
    LangChain's structured output / a Pydantic schema instead of a raw
    prompt) is likely more robust than adding more regular expressions here.

    Args:
        raw_response: The model's raw text output.

    Returns:
        A tuple (feedback_text, followup_text). followup_text is None if
        the model wrote "FOLLOWUP: NONE", left it empty, or the FOLLOWUP
        line is missing entirely.
    """
    text = normalize_model_response(raw_response)

    feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?:\n\s*FOLLOWUP:|\Z)', text, re.DOTALL | re.IGNORECASE)
    followup_match = re.search(r'FOLLOWUP:\s*(.*)', text, re.DOTALL | re.IGNORECASE)

    if feedback_match:
        feedback = _strip_leading_junk(feedback_match.group(1).strip())
    else:
        # The model didn't use the expected format — fail safe by treating
        # everything it said as the feedback, and assume no follow-up.
        feedback = text

    followup_raw = _strip_leading_junk(followup_match.group(1).strip()) if followup_match else ""
    no_followup_markers = ("NONE", "KEINE")
    if not followup_raw or followup_raw.upper().startswith(no_followup_markers):
        followup = None
    else:
        followup = followup_raw

    return feedback, followup


def format_aspects_context(question: dict) -> str:
    """
    Build an optional extra context block for behavior profiles that expect
    per-question "aspects" — content points a question is trying to
    capture, each with covered_if/maintenance_questions/followups (see e.g.
    study_content/interviews/nba_modul6_alltag_sozial.yaml and
    study_content/prompts/nba_conversational.txt).

    Returns "" for questions without an "aspects" field, so guides like
    depression_scid_example.yaml (which don't set it) get exactly the same
    prompt as before this was added. This is only ever extra CONTEXT for
    get_feedback_and_followup() below — the participant only ever sees or
    hears question["text"] itself, verbatim (see backend/app.py).

    Args:
        question: A question dict as built by InterviewGuide (may or may
            not have "aspects"/"example_if_unclear" set).

    Returns:
        A text block (with a leading newline) to insert into the prompt,
        or "" if the question has no aspects.
    """
    aspects = question.get("aspects")
    if not aspects:
        return ""

    lines = ["Aspects this question is trying to capture:"]
    for aspect in aspects:
        lines.append(f"- {aspect['name']} (covered_if: {aspect['covered_if']})")
        for maintenance_question in aspect.get("maintenance_questions") or []:
            lines.append(f'  maintenance_question: "{maintenance_question}"')
        for followup in aspect.get("followups") or []:
            lines.append(f'  followup: "{followup}"')

    example_if_unclear = question.get("example_if_unclear")
    if example_if_unclear:
        lines.append(f'example_if_unclear: "{example_if_unclear}"')

    return "\n" + "\n".join(lines) + "\n"


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower()).rstrip('?.!,;: ')


# Below this similarity ratio (difflib's SequenceMatcher, on normalized
# text), a model-generated followup is considered too different from
# anything in the question's own checklist to trust — see
# constrain_followup_to_checklist() below.
CHECKLIST_SIMILARITY_THRESHOLD = 0.6


def constrain_followup_to_checklist(question: dict, followup_text: str) -> str:
    """
    For questions with an "aspects" checklist, only ever let a followup
    reach the participant if it's (close to) one of the researcher-written
    maintenance_questions/followups for this question — never text the
    model invented itself. This extends the project's core principle "the
    LLM presents, it does not invent" (see interview_guide.py's module
    docstring, which states this for fixed questions) to checklist-based
    followups too, now that a fixed checklist exists to check against.

    Why this exists: smaller local models don't always follow "use one of
    the followups below" reliably — observed failure mode was the model
    inventing its own question that closely echoed back the participant's
    own words (e.g. participant asks "Was meinen Sie mit Pause?" and the
    model responds with a "followup" that just rephrases that same
    question back at them, instead of answering it or picking a real
    checklist item). That is jarring and must never reach the participant.

    Guides without "aspects" (e.g. depression_scid_example.yaml) are
    completely unaffected — this returns followup_text unchanged for them,
    exactly as before this function existed.

    Args:
        question: The current question dict (may or may not have
            "aspects" set).
        followup_text: The model's raw followup text (already known to not
            be None, SAFETY_CONCERN_MARKER, or REPEAT_QUESTION_MARKER —
            those are handled separately in backend/app.py).

    Returns:
        followup_text unchanged if the question has no aspects; the
        matching checklist item's exact wording if one is close enough
        (ratio >= CHECKLIST_SIMILARITY_THRESHOLD); otherwise a safe,
        generic fallback (the first aspect's first maintenance_question —
        content-neutral by design, so it can never itself echo back
        anything the participant said).
    """
    aspects = question.get("aspects")
    if not aspects:
        return followup_text

    candidates = []
    for aspect in aspects:
        candidates.extend(aspect.get("maintenance_questions") or [])
        candidates.extend(aspect.get("followups") or [])
    if not candidates:
        return followup_text

    target = _normalize_for_comparison(followup_text)
    best_candidate, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, _normalize_for_comparison(candidate), target).ratio()
        if ratio > best_ratio:
            best_candidate, best_ratio = candidate, ratio

    if best_ratio >= CHECKLIST_SIMILARITY_THRESHOLD:
        return best_candidate

    fallback_candidates = aspects[0].get("maintenance_questions") or aspects[0].get("followups") or candidates
    return fallback_candidates[0]


def escalate_to_specific_followup(question: dict, current_followup_text: str) -> str:
    """
    If current_followup_text is one of the question's generic
    maintenance_questions, switch to the first specific followup from the
    SAME aspect instead — used when the participant asked for
    clarification about a generic maintenance_question (e.g. "Was meinen
    Sie mit 'sonst noch etwas'?"). Repeating the same generic phrasing
    after explaining it doesn't actually help the participant: it was the
    generic abstraction itself that confused them, not a one-off wording
    issue, so a concrete, content-specific question is what should follow.

    Returns current_followup_text unchanged if it's already a specific
    followup (not a maintenance_question), or if the question has no
    aspects / the matching aspect has no followups to escalate to.

    Known limitation: if the SAME generic maintenance_question text is
    shared by more than one aspect on this question (as generic prompts
    like "Gibt es sonst noch etwas?" often are), this picks the first
    matching aspect in guide order — which may not be the one the model
    actually meant. Still strictly better than repeating the same generic
    text a second time, but not guaranteed to pick the "right" aspect.

    Args:
        question: The current question dict.
        current_followup_text: The pending followup text about to be
            re-presented after a clarification request (see
            backend/app.py's _handle_clarification_request()).
    """
    aspects = question.get("aspects")
    if not aspects:
        return current_followup_text

    target = _normalize_for_comparison(current_followup_text)
    for aspect in aspects:
        maintenance_questions = aspect.get("maintenance_questions") or []
        if any(_normalize_for_comparison(mq) == target for mq in maintenance_questions):
            followups = aspect.get("followups") or []
            if followups:
                return followups[0]
            break

    return current_followup_text


def format_required_style_context(style_index: int) -> str:
    """
    Build the REQUIRED_STYLE context line that tells the model exactly
    which FEEDBACK style to use this turn (1 = Spiegeln, 2 = Kleine echte
    Reaktion, 3 = Konkretes Aufgreifen — see
    study_content/prompts/nba_conversational.txt's "FEEDBACK styles"
    section).

    Style ROTATION is computed here, driven by backend/app.py's
    _style_turn_count, rather than left to the model to track itself.
    Earlier versions asked the model to "rotate between 3 styles, never
    twice in a row" — observed failure: the model ignored this and fell
    into a single repeated template instead (e.g. always "Klingt nach
    ..."), regardless of what was actually said. Telling it exactly which
    style number to use each turn removes that judgment call (and that
    failure mode) entirely — this is far more reliable than expecting an
    8B-class local model to self-track rotation across many turns.

    Args:
        style_index: 1, 2, or 3.

    Returns:
        A text block (with a leading newline) to insert into the prompt.
    """
    return f"\nREQUIRED_STYLE: {style_index}\n"


def get_feedback_and_followup(
    chain_with_history,
    profile,
    question_text: str,
    answer_text: str,
    followup_allowed: bool,
    language_code: str = "de",
    aspects_context: str = "",
    required_style_context: str = "",
) -> tuple[str, str | None]:
    """
    Ask the model for a short, neutral acknowledgement of the participant's
    answer, and — only if permitted for this specific item — an optional
    single follow-up question. This does NOT generate the next fixed
    question; that always comes verbatim from the interview guide (see
    backend/interview_guide.py) and is combined afterwards by the caller
    (see backend/app.py).

    Args:
        chain_with_history: The pipeline created by build_chain().
        profile: A BehaviorProfile (see backend/behavior_profiles.py)
            supplying the actual tone/strictness instructions.
        question_text: The fixed question (or pending follow-up) the
            participant was just asked.
        answer_text: What the participant said or typed in response.
        followup_allowed: Whether a follow-up is allowed for this specific
            item (set by the "allow_followup" field on the question, and
            only if one hasn't already been used for it — see
            InterviewGuide.followup_already_used()).
        language_code: The study's interview language (from
            study_settings.yaml, e.g. "de"). The FEEDBACK/FOLLOWUP text is
            explicitly required to be in this language — this is not left
            to the model to infer, since smaller local models are not
            reliable at that and can drift into English.
        aspects_context: Optional extra context block from
            format_aspects_context() above, for profiles that use
            per-question "aspects" (e.g. nba_conversational). Defaults to
            "" — guides without this field are unaffected.
        required_style_context: Optional extra context block from
            format_required_style_context() above, telling the model
            exactly which FEEDBACK style to use this turn. Defaults to "".

    Returns:
        A tuple (feedback_text, followup_text_or_None). See
        parse_feedback_and_followup() for how followup_text is determined.
    """
    # A single fixed session id is used because, per the project's core
    # principles, only one participant runs through the study at a time.
    session_id = "voice_assistant_session"

    followup_status = (
        "permitted for this item — you may use at most one, only if it "
        "would meaningfully help"
        if followup_allowed else
        "not permitted for this item — always write FOLLOWUP: NONE, unless "
        "your instructions define a separate marker for the participant "
        "asking a clarification question instead of answering (e.g. "
        "[REPEAT_QUESTION]) — that marker is about a different situation "
        "and remains available regardless of this setting"
    )

    language_name = _language_name(language_code)
    instructions = (
        f"{profile.instructions}\n\n"
        f"LANGUAGE (mandatory, overrides anything above): write both the "
        f"FEEDBACK text and the FOLLOWUP question (if used) in {language_name}, "
        f"regardless of what language these instructions themselves are "
        f"written in."
    )

    raw_response = chain_with_history.invoke(
        {
            "instructions": instructions,
            "question": question_text,
            "aspects_context": aspects_context,
            "required_style_context": required_style_context,
            "answer": answer_text,
            "followup_status": followup_status,
        },
        config={"session_id": session_id},
    )
    return parse_feedback_and_followup(raw_response)


def get_natural_transition(llm, feedback_text: str, next_question_text: str, language_code: str = "de") -> str:
    """
    For behavior profiles with question_delivery = "natural_transition":
    smoothly reword the transition from the just-given feedback into the
    next fixed question, without changing what the question actually asks.

    This is a separate, stateless call (uses the plain `llm`, not the
    chat-history chain) since it's a wording task, not part of the
    interview conversation itself.

    Args:
        llm: The LLM object created by create_llm() (not the history chain).
        feedback_text: The feedback sentence just generated for the
            participant's previous answer.
        next_question_text: The next fixed question, verbatim from the
            interview guide — its meaning must be fully preserved.
        language_code: The study's interview language (from
            study_settings.yaml, e.g. "de") — explicitly enforced for the
            same reason as in get_feedback_and_followup() above.

    Returns:
        One combined, naturally-worded message.
    """
    chain = transition_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "feedback": feedback_text,
        "next_question": next_question_text,
        "language_name": _language_name(language_code),
    })
    return normalize_model_response(result)
