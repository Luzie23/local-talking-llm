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
from agents.diagnosis_agent import DiagnosisAnalysisDraft
from agents.navigation_agent import NavigationDecision, decide_next_action
from agents.question_agent import generate_clarification_question
from session_record import build_item_record
# EMSAG: 03.11.2026
# Import the longitudinal evidence analysis agent.
from agents.diagnosis_agent import DiagnosisAgent, analyze_evidence
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



def validate_decision(decision, guide):
    # EMSAG: RT01: Allow unlimited clarification, but respect explicit item-level restrictions.
    if decision.action in ("clarify", "rephrase"):
        question = guide.current_question()
        if question is not None and not question.get("allow_clarification", True):
            return NavigationDecision(
                action="skip",
                skip_reason="unresolved",
                reason="Clarification is disabled for this SCID item.",
            )

    if decision.action == "advance":
        return decision.model_copy(
            update={"next_step": guide.next_item_id()}
        )

    if decision.action == "branch":
        if (
            decision.next_step is None
            or not guide.has_item(decision.next_step)
        ):
            raise ValueError(
                f"Invalid branch target: {decision.next_step}"
            )

    return decision


def handle_turn(
    llm,
    guide,
    recorder,
    answer_text: str,
    acknowledge=None,
    authorized_historical_session_ids=None,
) -> TurnOutcome:  

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

    #EMSAG: RT07: what if user wants to end the conversation?
    # Explicit request to end the entire interview.
    normalized_answer = " ".join(
        re.sub(r"[^\w\s]", " ", answer_text.casefold()).split()
    )

    stop_phrases = {
        "de": {
            "ich möchte das gespräch beenden",
            "ich will das gespräch beenden",
            "ich möchte aufhören",
            "ich möchte das interview beenden",
            "beende das gespräch",
        },
        "en": {
            "i want to end the conversation",
            "i want to end the interview",
            "i would like to stop",
            "please end the conversation",
        },
    }

    if normalized_answer in stop_phrases.get(guide.language, set()):
        guide.record_participant_turn(answer_text)
        guide._move_to(None)
        recorder.save()

        return TurnOutcome(
            "Natürlich. Wir beenden das Gespräch hier. Vielen Dank für Ihre Zeit."
            if guide.language == "de"
            else "Of course. We will end the conversation here. Thank you for your time.",
            "",
            "closing",
            {"action": "end_interview"},
        )
    #EMSAG: RT07: what if user wants to end the conversation?
    language = LANGUAGE_NAMES.get(guide.language, "German")
    guide.record_participant_turn(answer_text)

    # 1. Deterministic skip command: honoured even if the model is down.
    # EMSAG: 03.11.2026
    # Initialize diagnosis trace fields for both normal answers and deterministic skips.
    judgment = None
    diagnosis_result = None
    diagnosis_error = None
    if is_explicit_skip(answer_text, guide.skip_phrases):
        decision = NavigationDecision(action="skip", skip_reason="refused",
                                      reason="Explicit skip command (deterministic check).")
    else:
        # 2. Judgment.
        try:
            judgment = judge_answer(llm, item, guide.item_turns)
        except JudgmentError as exc:
            return _technical_failure(guide, recorder, item, str(exc))
        # EMSAG: 03.11.2026
        # Persist each Judgment Agent assessment immediately, including assessments before clarification or navigation.
        recorder.record_judgment(item["id"], judgment)

        # EMSAG: 03.11.2026
        # Analyze current and authorized longitudinal evidence without controlling interview navigation.
        try:
            if not recorder.session_id:
                raise RuntimeError("No active database session is available.")

            #diagnosis_agent = DiagnosisAgent()

           # evidence_summaries = diagnosis_agent.collect_evidence(
            #    current_session_id=recorder.session_id,
            #    guide=guide,
            #    authorized_historical_session_ids=(
            #        authorized_historical_session_ids or []
            #    ),
            #    current_item_id=item["id"],
            #    current_item_turns=guide.item_turns,
            #)

            #diagnosis_draft = analyze_evidence(
            #    llm=llm,
            #    evidence_summaries=evidence_summaries,
            #    guide=guide,
            #)

            #diagnosis_result = diagnosis_draft
            diagnosis_result = None
            diagnosis_error = None

        except Exception as exc:
            diagnosis_error = str(exc)
            print(f"[diagnosis_agent] Analysis failed: {diagnosis_error}")
        # 3. Navigation proposal.
        # EMSAG: Select the counter matching the participant's response type.
        if judgment.intent in ("off_topic", "unclear"):
            interaction_kind = "redirect"
        elif judgment.intent == "clarification_request":
            interaction_kind = "rephrase"
        elif judgment.intent == "cannot_answer":
            interaction_kind = "gentle"
        else:
            interaction_kind = "clarify"

        decision = decide_next_action(
            intent=judgment.intent,
            sufficiency=judgment.sufficiency,
            clarification_count=guide.current_interaction_count(interaction_kind),
            max_clarifications=guide.max_clarifications(),
            next_step=item.get("next_step"),
            cannot_answer_moves_on=item.get("cannot_answer_moves_on", False),
        )

    guide.consecutive_system_errors = 0

    # 4. Validation of the proposal.
    proposed = decision
    decision = validate_decision(decision, guide)
    # EMSAG: 03.11.2026
    # Keep Diagnosis Agent output available for research tracing without changing navigation.

    trace = {
        "item": item["id"],
        "judgment": judgment.model_dump() if judgment else None,
        "diagnosis": ( diagnosis_result.model_dump() if diagnosis_result is not None else None),
        "diagnosis_error": diagnosis_error,
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
        # EMSAG: Changed fallback handling to prevent valid Question Agent responses
        # from being overwritten by hardcoded text.
        if text is None:
            # Deterministic fallback when the Question Agent fails twice.
            recorder.log_system_event(
                item["id"],
                "question_agent_fallback",
                "invalid or failed output",
            )
            #EMSAG: RT02: "anders gefragt:" ist not used anymore
            if decision.question_mode == "rephrase":
                text = (
                    "Ich meine, ob Sie die Frage in Ihren eigenen Worten "
                    "beschreiben können. Was bedeutet das für Sie?"
                    if guide.language == "de"
                    else "I mean, could you describe the question in your own words? "
                        "What does it mean to you?"
                )
            elif decision.question_mode == "redirect":
                text = (
                    "Lassen Sie uns zu meiner Frage zurückkommen: "
                    f"{item['text']}"
                    if guide.language == "de"
                    else "Let us return to my question: "
                        f"{item['text']}"
                )
            elif decision.question_mode == "gentle":
                text = (
                    "Sie können sich gern etwas Zeit nehmen. "
                    "Was fällt Ihnen dazu ein?"
                    if guide.language == "de"
                    else "Take your time. What comes to mind?"
                )
            else:
                text = (
                    item["text"]
                )
            #EMSAG: RT02: "anders gefragt:" ist not used anymore
            trace["question_fallback"] = True
        else:
            trace["question_fallback"] = False

        print(
            f"[orchestrator] mode={decision.question_mode}, "
            f"fallback={trace['question_fallback']}, response={text!r}"
        )

        # EMSAG: Changed fallback handling to prevent valid Question Agent responses
        # from being overwritten by hardcoded text.
        # EMSAG: 03.11.2026
        # Record the validated clarification decision; no SCID item transition has occurred.
        guide.start_clarification(text, kind=decision.question_mode or decision.action)

        recorder.record_navigation(
            item_id=item["id"],
            proposed=proposed,
            validated=decision,
            executed_next_item_id=None,
        )

        recorder.save()
        return TurnOutcome("", text, "question", trace)

    clarifications_used = guide.current_clarification_count()
    if decision.action in ("advance", "branch"):
        record = build_item_record(item, list(guide.item_turns), "answered", judgment=judgment,
                                   decision=decision, clarifications_used=clarifications_used)
        feedback = _acknowledgement(acknowledge, guide, answer_text)
        next_item = guide.finish_item(record, decision.next_step)
    #EMSAG: RT04: Skip Refusal Conditions were improved
    else:  # skip
        record = build_item_record(
            item,
            list(guide.item_turns),
            "skipped",
            judgment=judgment,
            skip_reason=decision.skip_reason,
            decision=decision,
            clarifications_used=clarifications_used,
        )

        if decision.skip_reason == "refused":
            feedback = (
                "Das ist völlig in Ordnung. "
                "Wir können zur nächsten Frage übergehen."
                if guide.language == "de"
                else "That's completely fine. "
                    "We can move on to the next question."
            )
        else:
            feedback = _acknowledgement(
                acknowledge, guide, answer_text
            )

        next_item = guide.finish_item(record, guide.next_item_id())
    #EMSAG: RT04: Skip Refusal Conditions were improved

    # EMSAG: 03.11.2026
    # Record the actual successor returned by InterviewGuide after the validated transition executes.
    recorder.record_navigation(
        item_id=item["id"],
        proposed=proposed,
        validated=decision,
        executed_next_item_id=next_item["id"] if next_item is not None else None,
    )

    recorder.save()
    if next_item is None:
        return TurnOutcome(feedback, guide.closing_message, "closing", trace)
    return TurnOutcome(feedback, guide.presentation_text(next_item), "question", trace)


def _default_prefix(guide) -> str:
    """Lead-in for the deterministic fallback reformulation (used only when the
    Question Agent fails twice): the original item is shown again with this prefix."""
    return "Anders gefragt:" if guide.language == "de" else "To put it differently:"

#EMSAG: RT04: not only saying danke but also give more empathy and acknowladgement
def _acknowledgement(acknowledge, guide, answer_text: str) -> str:
    """Return a meaningful acknowledgement without generic thanks alone."""
    if acknowledge is None:
        return guide.messages["acknowledge"]

    try:
        text = (acknowledge(guide.active_prompt_text(), answer_text) or "").strip()
    except Exception:
        text = ""

    # An acknowledgement must not contain a follow-up question.
    if not text or "?" in text:
        text = ""

    # Reject generic thanks as the entire acknowledgement.
    normalized = " ".join(text.casefold().split()).strip(" .!,:;")
    generic_thanks = {
        "danke",
        "vielen dank",
        "danke für ihre antwort",
        "danke für ihre offenheit",
        "thank you",
        "thanks",
        "thank you for your answer",
        "thanks for sharing",
    }

    if normalized in generic_thanks:
        text = ""

    if text:
        return text

    # Use a context-sensitive fallback for clearly emotional disclosures.
    answer_lower = answer_text.casefold()
    emotional_terms = (
        "einsam", "allein", "niemand", "traurig", "tod",
        "gestorben", "hoffnungslos", "wertlos", "misshandelt",
        "gewalt", "verletzt", "verzweifelt",
    )

    if any(term in answer_lower for term in emotional_terms):
        return "Es klingt belastend, was Sie beschreiben."

    # Preserve the existing fallback for other answers.
    return guide.messages["acknowledge"]
#EMSAG: RT04: not only saying danke but also give more empathy and acknowladgement


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
