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

    # Remove a leading comma/space from replies like ", Hallo..."
    normalized = re.sub(r'^[,\s\u2028\u2029]+', '', normalized)
    return normalized.strip()


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
        feedback = feedback_match.group(1).strip()
    else:
        # The model didn't use the expected format — fail safe by treating
        # everything it said as the feedback, and assume no follow-up.
        feedback = text

    followup_raw = followup_match.group(1).strip() if followup_match else ""
    no_followup_markers = ("NONE", "KEINE")
    if not followup_raw or followup_raw.upper().startswith(no_followup_markers):
        followup = None
    else:
        followup = followup_raw

    return feedback, followup


def get_feedback_and_followup(
    chain_with_history,
    profile,
    question_text: str,
    answer_text: str,
    followup_allowed: bool,
    language_code: str = "de",
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
        "not permitted for this item — always write FOLLOWUP: NONE"
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
