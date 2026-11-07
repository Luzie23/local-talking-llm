
"""
NAVIGATION AGENT
================
Deterministic interview navigation.

The agent proposes an action. The Orchestrator validates and executes it.
Interaction limits are tracked per item by the Orchestrator.

Policy:
- A participant who does not understand the question receives another
  explanation/rephrasing; this intent alone never causes an automatic skip.
- Off-topic or unclear replies get one redirect; a second unsuccessful
  attempt skips the item.
- Cannot-answer replies get one gentle follow-up; a second one skips,
  unless the guide says to move on immediately.
- A refusal gets one non-pressuring offer to help or skip; a second refusal
  skips. An explicit skip command should still be honoured immediately
  by the Orchestrator.
- Sufficient answers advance normally.
"""
from typing import Literal

from pydantic import BaseModel

NavigationAction = Literal["advance", "branch", "clarify", "rephrase", "skip"]
SkipReason = Literal["refused", "cannot_answer", "unresolved"]
ClarifyMode = Literal["clarify", "gentle", "redirect", "rephrase"]


class NavigationDecision(BaseModel):
    """A proposed next action plus the rule that produced it."""

    action: NavigationAction
    reason: str
    next_step: str | None = None
    skip_reason: SkipReason | None = None
    question_mode: ClarifyMode | None = None




def decide_next_action(
    intent: str,
    sufficiency: str,
    clarification_count: int,
    max_clarifications: int,
    next_step: str | None = None,
    cannot_answer_moves_on: bool = False,
) -> NavigationDecision:
    """Propose the next action using response-type-specific attempt counts."""

    valid_intents = {
        "answer",
        "clarification_request",
        "cannot_answer",
        "refusal",
        "off_topic",
        "unclear",
    }
    valid_sufficiency = {"sufficient", "ambiguous", "insufficient"}

    if intent not in valid_intents:
        raise ValueError(f"Unknown intent: {intent!r}")
    if sufficiency not in valid_sufficiency:
        raise ValueError(f"Unknown sufficiency: {sufficiency!r}")

    # EMSAG: Sufficient answers advance normally.
    if intent == "answer" and sufficiency == "sufficient":
        return NavigationDecision(
            action="advance",
            next_step=next_step,
            reason="Enough information to assess the item.",
        )

    # EMSAG: Rephrase requests are unlimited.
    if intent == "clarification_request":
        return NavigationDecision(
            action="rephrase",
            question_mode="rephrase",
            reason="Participant requests an explanation or reformulation.",
        )

    # EMSAG: Explicit refusal skips immediately.
    if intent == "refusal":
        return NavigationDecision(
            action="skip",
            skip_reason="refused",
            reason="Participant explicitly declined to answer.",
        )

    # EMSAG: Respect source-defined SCID cannot-answer instructions.
    if intent == "cannot_answer":
        if cannot_answer_moves_on:
            return NavigationDecision(
                action="skip",
                skip_reason="cannot_answer",
                reason="Source instruction requires moving on when participant cannot answer.",
            )

        if clarification_count >= 1:
            return NavigationDecision(
                action="skip",
                skip_reason="cannot_answer",
                reason="Participant still cannot answer after one gentle follow-up.",
            )

        return NavigationDecision(
            action="clarify",
            question_mode="gentle",
            reason="First cannot-answer response; offer one gentle follow-up.",
        )

    # EMSAG: Off-topic/unclear gets one redirect.
    if intent in ("off_topic", "unclear"):
        if clarification_count >= 1:
            return NavigationDecision(
                action="skip",
                skip_reason="unresolved",
                reason="Second unsuccessful off-topic/unclear response.",
            )

        return NavigationDecision(
            action="clarify",
            question_mode="redirect",
            reason="First off-topic/unclear response; redirect to the item.",
        )

    # EMSAG: Ambiguous/insufficient gets one clarification.
    if clarification_count >= 1:
        return NavigationDecision(
            action="skip",
            skip_reason="unresolved",
            reason="Second unresolved answer; move on to the next item.",
        )

    return NavigationDecision(
        action="clarify",
        question_mode="clarify",
        reason=f"Answer is {sufficiency}; ask about the missing information.",
    )
