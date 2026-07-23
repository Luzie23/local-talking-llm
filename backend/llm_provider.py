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
study_content/interviews/*.yaml (see backend/interview_guide.py). Step 3
("Constrain the LLM") will replace the current placeholder instructions
below with the real, carefully worded behavior rules.

Used by: backend/app.py (the Flask web routes) and the terminal mode.
"""
import os
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import OllamaLLM


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


# The instructions given to the model before every conversation.
# NOTE: this is a placeholder for now. Step 3 of the project plan ("Constrain
# the LLM") will replace this with the real interview behavior rules, chosen
# per interview guide via its "behavior_profile" field. For now, one fixed,
# simple rule set is used for every guide.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are conducting a structured research interview. The participant "
        "was just asked a fixed, pre-written question and has now answered it. "
        "Give a short (max. about 15 words), neutral, empathetic acknowledgement "
        "of their answer. Do not ask a question of your own. Do not give a "
        "diagnosis or any clinical assessment. Always reply in the same "
        "language the participant used."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "I was asked: \"{question}\"\nI answered: \"{answer}\""),
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


def get_feedback_response(chain_with_history, question_text: str, answer_text: str) -> str:
    """
    Ask the model for a short, neutral acknowledgement of the participant's
    answer to a fixed question. This does NOT generate the next question —
    that always comes verbatim from the interview guide (see
    backend/interview_guide.py); this function only produces the brief
    reaction in between two fixed questions.

    Args:
        chain_with_history: The pipeline created by build_chain().
        question_text: The fixed question the participant was just asked.
        answer_text: What the participant said or typed in response.

    Returns:
        The model's short feedback, cleaned up and ready to display or
        speak aloud, with the next fixed question appended afterwards by
        the caller (see backend/app.py).
    """
    # A single fixed session id is used because, per the project's core
    # principles, only one participant runs through the study at a time.
    session_id = "voice_assistant_session"

    response = chain_with_history.invoke(
        {"question": question_text, "answer": answer_text},
        config={"session_id": session_id}
    )
    return normalize_model_response(response)
