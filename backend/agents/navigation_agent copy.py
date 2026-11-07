"""
NAVIGATION AGENT
==================
Proposes the next interview action from the Judgment Agent's result and the
current interview state. It is deliberately rule-based (no LLM): interview
flow must be predictable, testable and traceable.

Boundary: Navigation only PROPOSES. The Orchestrator
(interview_orchestrator.py) validates the proposal against the guide
(budget, existing transition targets) and executes it.

Actions:
  advance   – the item can be closed; go to its configured successor
  branch    – go to a configured branch target (reserved: the current guides
              define no answer-dependent branches; see README "open decisions")
  clarify   – ask ONE focused question about the missing information
              (also used for a gentle follow-up when the participant cannot
              answer, and for a neutral redirect after off-topic/unclear replies)
  rephrase  – briefly explain and reformulate the SAME item
  skip      – close the item WITHOUT a result, with a reason
              (refused / cannot_answer / unresolved). A skip is never a
              negative answer.

Decision table (project-specific interaction rules, not SCID rules):

  intent                sufficiency            budget left        budget used up
  --------------------  ---------------------  -----------------  ---------------------------
  refusal               (any)                  skip: refused      skip: refused
  answer                sufficient             advance            advance
  answer                ambiguous/insufficient clarify            skip: unresolved
  clarification_request (any)                  rephrase           skip: unresolved
  cannot_answer         (any)                  clarify (gentle)   skip: cannot_answer
  off_topic / unclear   (any)                  clarify (redirect) skip: unresolved
"""
from typing import Literal

from pydantic import BaseModel

NavigationAction = Literal["advance", "branch", "clarify", "rephrase", "skip"]
SkipReason = Literal["refused", "cannot_answer", "unresolved"]
ClarifyMode = Literal["clarify", "gentle", "redirect", "rephrase"]


class NavigationDecision(BaseModel):
    """A proposed next action plus the rule that produced it (for the trace)."""

    action: NavigationAction
    reason: str
    next_step: str | None = None
    skip_reason: SkipReason | None = None
    # Which kind of text the Question Agent should produce (clarify/rephrase only).
    question_mode: ClarifyMode | None = None


def decide_next_action(
    intent: str,
    sufficiency: str,
    clarification_count: int,
    max_clarifications: int,
    next_step: str | None = None,
    cannot_answer_moves_on: bool = False,
) -> NavigationDecision:
    """
    Propose the next action deterministically.

    Args:
        intent: Judgment intent label.
        sufficiency: Judgment sufficiency label.
        clarification_count: Clarifications already used for this item.
        max_clarifications: Configured budget for this item (0 = none).
        next_step: The item's configured successor (None = end / file order).
        cannot_answer_moves_on: Source rule for this item ("IF CAN'T ANSWER,
            MOVE ON", SCID-5-PD overview): no follow-up when the participant
            cannot answer.

    Returns:
        A NavigationDecision (a proposal; the Orchestrator validates it).

    Raises:
        ValueError: on unknown labels or negative counters, so that a bug
            can never silently turn into "advance".
    """
    valid_intents = {"answer", "clarification_request", "cannot_answer", "refusal", "off_topic", "unclear"}
    valid_sufficiency = {"sufficient", "ambiguous", "insufficient"}
    if intent not in valid_intents:
        raise ValueError(f"Unknown intent: {intent!r}")
    if sufficiency not in valid_sufficiency:
        raise ValueError(f"Unknown sufficiency: {sufficiency!r}")
    if clarification_count < 0 or max_clarifications < 0:
        raise ValueError("Clarification counters cannot be negative.")

    budget_left = clarification_count < max_clarifications

    # The participant's right to refuse comes first: skip immediately,
    # no explanation, no persuasion, no budget needed.
    if intent == "refusal":
        return NavigationDecision(action="skip", skip_reason="refused",
                                  reason="Participant declined or asked to skip.")

    if intent == "answer" and sufficiency == "sufficient":
        return NavigationDecision(action="advance", next_step=next_step,
                                  reason="Enough information to assess the item.")

    if intent == "cannot_answer" and cannot_answer_moves_on:
        return NavigationDecision(action="skip", skip_reason="cannot_answer",
                                  reason="Source rule for this item: if the participant can't answer, move on.")

    if not budget_left:
        reason = "cannot_answer" if intent == "cannot_answer" else "unresolved"
        return NavigationDecision(
            action="skip", skip_reason=reason,
            reason=f"Clarification budget used ({clarification_count}/{max_clarifications}); "
                   f"item could not be assessed (intent={intent}, sufficiency={sufficiency}).")

    if intent == "clarification_request":
        return NavigationDecision(action="rephrase", question_mode="rephrase",
                                  reason="Participant did not understand the question.")
    if intent == "cannot_answer":
        return NavigationDecision(action="clarify", question_mode="gentle",
                                  reason="Participant cannot answer yet; one gentle follow-up.")
    if intent in ("off_topic", "unclear"):
        return NavigationDecision(action="clarify", question_mode="redirect",
                                  reason=f"Reply was {intent}; neutral redirect to the item.")
    return NavigationDecision(action="clarify", question_mode="clarify",
                              reason=f"Answer is {sufficiency}; ask for the missing information.")
