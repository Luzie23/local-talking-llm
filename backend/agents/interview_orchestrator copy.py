"""
INTERVIEW ORCHESTRATOR
========================
Coordinates one participant turn. It is the ONLY component that changes the
interview state (InterviewGuide) — the agents only return proposals/text:

  participant text
     │
     ├─ 0. paused?  → fixed "paused" message (after repeated technical failures)
     ├─ 1. deterministic explicit-skip check (skip_phrases) → refusal, no LLM needed
     ├─ 2. Judgment Agent  → intent + sufficiency (+ provisional rating, evidence)
     │        └─ failure → technical-error path (no advance, no budget used)
     ├─ 3. Navigation Agent → PROPOSED action
     ├─ 4. validate proposal against the guide (budget, transition target)
     ├─ 5. execute:
     │        clarify/rephrase → Question Agent text (validated, fallback) and
     │                           one unit of the clarification budget
     │        advance          → item record "answered", short acknowledgement,
     │                           configured successor
     │        skip             → item record "skipped" + reason, fixed neutral
     │                           message, configured successor
     └─ 6. save the session record

Safety: the keyword safety check runs in app.py BEFORE this function on every
message and is independent of everything here; refusal/clarification handling
can therefore never disable it.

Model/validation failures are kept apart from participant behaviour: they
are logged as system events, the participant is asked to resend, the
clarification budget is not used, and after MAX_CONSECUTIVE_SYSTEM_ERRORS the
interview pauses instead of skipping content.
"""
import re
from dataclasses import dataclass, field

from agents.judgment_agent import JudgmentError, judge_answer
from agents.navigation_agent import NavigationDecision, decide_next_action
from agents.question_agent import generate_clarification_question
from session_record import build_item_record

MAX_CONSECUTIVE_SYSTEM_ERRORS = 3

LANGUAGE_NAMES = {"de": "German", "en": "English"}


@dataclass
class TurnOutcome:
    """What the application should show after a turn.

    feedback:   acknowledgement / skip notice / error message (may be "")
    next_text:  the next prompt (item, clarification or closing), or None
    next_kind:  "question" | "closing" | None (used for "repeat question")
    trace:      decision details for the console log
    """
    feedback: str
    next_text: str | None
    next_kind: str | None
    trace: dict = field(default_factory=dict)


def is_explicit_skip(answer_text: str, skip_phrases: list[str]) -> bool:
    """
    Deterministic detection of an explicit skip command such as
    "überspringen" or "Bitte überspringen." (exact phrase, optionally with
    "bitte"/"please"). Kept strict on purpose so that ordinary answers
    containing these words are not mistaken for a skip; anything less
    explicit is left to the Judgment Agent (intent "refusal").
    """
    text = re.sub(r"[^\w\s]", " ", answer_text.lower())
    text = re.sub(r"\b(bitte|please|einfach|ich möchte|i want to|let's|lass uns)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text in skip_phrases


def validate_decision(decision: NavigationDecision, guide) -> NavigationDecision:
    """
    Check a proposed action against the current state and configuration
    and return the action that is actually permitted.

    * clarify/rephrase is only permitted while budget remains; otherwise the
      item is closed as unresolved (never silently advanced).
    * advance/branch must target an existing item or the end; the configured
      successor of the item is used for advance.
    """
    if decision.action in ("clarify", "rephrase") and guide.clarification_limit_reached():
        return NavigationDecision(action="skip", skip_reason="unresolved",
                                  reason="Validation: clarification budget exhausted; " + decision.reason)
    if decision.action == "advance":
        return decision.model_copy(update={"next_step": guide.next_item_id()})
    if decision.action == "branch":
        if decision.next_step is None or not guide.has_item(decision.next_step):
            raise ValueError(f"Branch target {decision.next_step!r} does not exist in the guide.")
    return decision


def handle_turn(llm, guide, recorder, answer_text: str, acknowledge=None) -> TurnOutcome:
    """
    Process one participant message for the current item.

    Args:
        llm: The model used by the Judgment and Question Agents.
        guide: InterviewGuide (state; changed only here).
        recorder: SessionRecorder (researcher-only JSON record).
        answer_text: The participant's message.
        acknowledge: Optional callable (prompt_text, answer_text) -> str that
            produces a short acknowledgement after a sufficient answer (the
            existing behavior-profile feedback chain). Falls back to a fixed text.

    Returns:
        TurnOutcome for the application to display.
    """
    item = guide.current_question()
    if item is None:
        return TurnOutcome("", guide.closing_message, "closing", {"action": "finished"})
    if guide.paused:
        return TurnOutcome(guide.messages["paused"], None, None, {"action": "paused"})

    language = LANGUAGE_NAMES.get(guide.language, "German")
    guide.record_participant_turn(answer_text)

    # 1. Deterministic skip command: honoured even if the model is down.
    judgment = None
    if is_explicit_skip(answer_text, guide.skip_phrases):
        decision = NavigationDecision(action="skip", skip_reason="refused",
                                      reason="Explicit skip command (deterministic check).")
    else:
        # 2. Judgment.
        try:
            judgment = judge_answer(llm, item, guide.item_turns)
        except JudgmentError as exc:
            return _technical_failure(guide, recorder, item, str(exc))
        # 3. Navigation proposal.
        decision = decide_next_action(
            intent=judgment.intent,
            sufficiency=judgment.sufficiency,
            clarification_count=guide.current_clarification_count(),
            max_clarifications=guide.max_clarifications(),
            next_step=item.get("next_step"),
            cannot_answer_moves_on=item.get("cannot_answer_moves_on", False),
        )
    guide.consecutive_system_errors = 0

    # 4. Validation of the proposal.
    proposed = decision
    decision = validate_decision(decision, guide)
    trace = {
        "item": item["id"],
        "judgment": judgment.model_dump() if judgment else None,
        "proposed": proposed.model_dump(),
        "executed": decision.model_dump(),
        "clarifications_used": guide.current_clarification_count(),
    }

    # 5. Execution.
    if decision.action in ("clarify", "rephrase"):
        text = generate_clarification_question(
            llm, item, guide.item_turns,
            judgment.missing_information if judgment else [],
            mode=decision.question_mode or "clarify", language=language,
        )
        if text is None:
            # Deterministic fallback when the Question Agent fails twice.
            recorder.log_system_event(item["id"], "question_agent_fallback", "invalid or failed output")
            text = f"{guide.messages.get('rephrase_prefix', _default_prefix(guide))} {item['text']}"
            trace["question_fallback"] = True
        guide.start_clarification(text, kind=decision.question_mode or decision.action)
        recorder.save()
        return TurnOutcome("", text, "question", trace)

    clarifications_used = guide.current_clarification_count()
    if decision.action in ("advance", "branch"):
        record = build_item_record(item, list(guide.item_turns), "answered", judgment=judgment,
                                   decision=decision, clarifications_used=clarifications_used)
        feedback = _acknowledgement(acknowledge, guide, answer_text)
        next_item = guide.finish_item(record, decision.next_step)
    else:  # skip
        record = build_item_record(item, list(guide.item_turns), "skipped", judgment=judgment,
                                   skip_reason=decision.skip_reason, decision=decision,
                                   clarifications_used=clarifications_used)
        feedback = guide.messages[f"skip_{decision.skip_reason}"]
        next_item = guide.finish_item(record, guide.next_item_id())

    recorder.save()
    if next_item is None:
        return TurnOutcome(feedback, guide.closing_message, "closing", trace)
    return TurnOutcome(feedback, guide.presentation_text(next_item), "question", trace)


def _default_prefix(guide) -> str:
    """Lead-in for the deterministic fallback reformulation (used only when the
    Question Agent fails twice): the original item is shown again with this prefix."""
    return "Anders gefragt:" if guide.language == "de" else "To put it differently:"


def _acknowledgement(acknowledge, guide, answer_text: str) -> str:
    """Short acknowledgement after a sufficient answer; fixed text if the model fails."""
    if acknowledge is None:
        return guide.messages["acknowledge"]
    try:
        text = (acknowledge(guide.active_prompt_text(), answer_text) or "").strip()
    except Exception:
        text = ""
    # The acknowledgement must not ask anything: the next item follows directly.
    if not text or "?" in text:
        return guide.messages["acknowledge"]
    return text


def _technical_failure(guide, recorder, item, detail: str) -> TurnOutcome:
    """
    Handle a failed/malformed model response: do not advance, do not spend
    budget, do not keep the unjudged message as evidence; ask to resend;
    pause after repeated failures.
    """
    guide.item_turns.pop()  # the message will be judged again when resent
    guide.consecutive_system_errors += 1
    recorder.log_system_event(item["id"], "judgment_failure", detail)
    if guide.consecutive_system_errors >= MAX_CONSECUTIVE_SYSTEM_ERRORS:
        guide.paused = True
        recorder.save()
        return TurnOutcome(guide.messages["paused"], None, None, {"action": "paused", "detail": detail})
    recorder.save()
    return TurnOutcome(guide.messages["system_error"], guide.active_prompt_text(), "question",
                       {"action": "system_error", "detail": detail})
