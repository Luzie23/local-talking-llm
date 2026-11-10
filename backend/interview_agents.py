"""
INTERVIEW AGENTS — THE AGENT INTERFACES AND THE PHASE 2.5 STUBS
================================================================
RECONSTRUCTION_PLAN.md, Phase 2, step 2.5 — "Supervisor skeleton with
stub agents".

The pipeline shape (fixed before any real LLM is wired in, Phase 3):

    Interventionsmonitor -> abort check -> Diagnostiker -> Navigator
    -> Sprecher -> Validator -> Output

The Navigator (backend/interview_navigator.py) is a deterministic tool,
not an agent — it has no interface here. The other components are
pluggable behind the four agent protocols below; the deterministic
stubs are the DEFAULT wiring of the Supervisor:

    * PassthroughMonitor — the pass-through Monitor stub: it checks the
      input and NEVER fires a signal (Phase 4 wires the keyword/LLM
      detection of the three separate typed signals).
    * StubDiagnostiker   — the Diagnostiker stub: "answer recorded, no
      criterion change" — it reports the answer CLEAR, extracts no
      navigation values, proposes no criterion updates and makes no
      next-question choice (the deterministic fallback ordering applies;
      Phase 3 wires the LLM that evaluates the answer as far as the
      steering decision requires).
    * VerbatimSprecher   — the verbatim Sprecher stub: it returns the
      released content UNCHANGED (mode=VERBATIM); Phase 3 wires the
      LLM that reformulates without changing diagnostically relevant
      meaning.
    * AllowAllValidator  — the allow-all Validator stub: it releases
      every version (ALLOW); Phase 4 wires the binding exit control
      (meaning fidelity, transparency, role conformance, non-diagnosis,
      patient safety — fail-closed).

The OUTPUT layer is a pluggable port (OutputPort): in this step the
default InMemoryOutputPort only records what the pipeline handed over
(Phase 5 wires the real delivery protocol: reservation, ACK,
retransmission — the modality adapters themselves are untouched).

All stubs are PURE and deterministic: no LLM, no I/O. Their outputs are
the contracts' boundary types (contracts.py); the Supervisor applies
them (it remains the sole writer of the run state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contracts import (
    AnswerClarity,
    ApprovedContentPackage,
    ContentPiece,
    InputIntent,
    MonitorResult,
    Modality,
    NavigationAssessment,
    QuestionCandidateSet,
    SpeakerMode,
    SpeakerOutput,
    StudyCondition,
    ValidationAspect,
    ValidationDecision,
    ValidationOutcome,
)


# ---------------------------------------------------------------------------
# the delivery handed to the output layer (one released content piece)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Delivery:
    """
    One released piece of participant-facing content the Supervisor
    hands to the output layer. `delivery_id` is the delivery protocol's
    identifier (contracts.delivery_id: <run_id>__dNNN); the output
    layer ACKs THIS id when the participant has seen/heard the content
    (Phase 5). `question_id` is the question the delivery delivers —
    None for terminal content (closing, abort closing, out-of-scope
    hand-over, safety hint).
    """

    run_id: str
    delivery_id: str
    turn_id: str | None
    modality: Modality
    content: ContentPiece
    question_id: str | None = None


# ---------------------------------------------------------------------------
# the agent protocols (the interfaces the real LLM agents implement)
# ---------------------------------------------------------------------------


@runtime_checkable
class InterventionsMonitor(Protocol):
    """
    The Interventionsmonitor: checks EVERY participant input BEFORE
    regular interview processing. Returns a MonitorResult with the
    fired signals as the three SEPARATE typed signals (SuicidalitySignal
    / AbortRequestSignal / PromptInjectionSignal — never merged into one
    safety score; an empty signal tuple means nothing fired) AND the
    input's INTENT (contracts.InputIntent — WInf-inspired: the intent
    measurement lives HERE, because the monitor is the component that
    sees every input before the regular processing; the Diagnostiker
    only evaluates an answer against the delivered question). The
    intent drives the Supervisor's deterministic steering reactions
    (interview_navigator.steering_decision) — clarification_request /
    cannot_answer / refusal / off_topic / unclear, the default ANSWER.
    """

    def check(
        self, run_id: str, turn_id: str, transcript: str
    ) -> MonitorResult:
        """Check one participant input (the typed text or the transcript)."""
        ...


@runtime_checkable
class Diagnostiker(Protocol):
    """
    The Diagnostiker: evaluates an answer only as far as the steering
    decision requires — answer clarity, the navigation-relevant values,
    and (optionally) the choice of the contextually best next question.
    Its `next_question_id` must be one of the candidates of the
    QuestionCandidateSet handed over — the restriction is enforced by
    the pipeline (the Supervisor), not by this interface.
    """

    def assess(
        self,
        run_id: str,
        turn_id: str,
        question_id: str,
        transcript: str,
        candidates: QuestionCandidateSet,
    ) -> NavigationAssessment:
        """Evaluate one participant answer for the delivered question."""
        ...


@runtime_checkable
class Sprecher(Protocol):
    """
    The Sprecher: reformulates the content released by the Diagnostiker
    and Navigator to sound more natural without changing diagnostically
    relevant meaning. `revision_instructions` carries the Validator's
    bounded-revision instructions (None on the first round).
    """

    def speak(
        self,
        package: ApprovedContentPackage,
        revision_instructions: str | None = None,
    ) -> SpeakerOutput:
        """Produce the spoken version of one approved content package."""
        ...


@runtime_checkable
class Validator(Protocol):
    """
    The semantic Validator: the binding exit control on the
    Sprecher's version (meaning fidelity, transparency, role
    conformance, non-diagnosis, patient safety). ALLOW / REVISE /
    BLOCK; fail-closed — any internal error is a BLOCK.
    """

    def validate(
        self, output: SpeakerOutput, condition: StudyCondition | None = None
    ) -> ValidationDecision:
        """
        Decide on one SpeakerOutput (the study condition is the
        transparency/role constraint to check against).
        """
        ...


@runtime_checkable
class OutputPort(Protocol):
    """
    The OUTPUT layer's port: receives the released content and gets it
    to the participant (text / TTS / avatar). In step 2.5 the default
    implementation only records the delivery; the delivery protocol
    (reservation, ACK, retransmission) arrives with Phase 5.
    """

    def deliver(self, delivery: Delivery) -> None:
        """Hand over one released delivery (the server-side send)."""
        ...

# ---------------------------------------------------------------------------
# the deterministic stubs (the default wiring, step 2.5)
# ---------------------------------------------------------------------------


class PassthroughMonitor:
    """
    The pass-through Monitor stub: it performs the check but NEVER
    fires a signal and classifies EVERY input as ANSWER — the
    pipeline's intervention shape (the typed signals, the abort check,
    the safety reactions) and the steering shape (the intent-driven
    reactions) both stay exercisable by injection, while the default
    flow is unobstructed (the steering table degrades to the classic
    clarity-only path). Phase 4 wires the real detection: the keyword /
    LLM signal layers AND the intent classification (WInf-inspired).
    """

    def check(
        self, run_id: str, turn_id: str, transcript: str
    ) -> MonitorResult:
        return MonitorResult(
            signals=(), intent=InputIntent.ANSWER
        )


class StubDiagnostiker:
    """
    The Diagnostiker stub: "answer recorded, no criterion change". It
    reports every answer CLEAR, extracts no navigation values,
    proposes no criterion updates and makes no next-question choice —
    the interview advances via the Navigator's deterministic fallback
    ordering (postorder + the definition's jumps).
    """

    def assess(
        self,
        run_id: str,
        turn_id: str,
        question_id: str,
        transcript: str,
        candidates: QuestionCandidateSet,
    ) -> NavigationAssessment:
        return NavigationAssessment(
            run_id=run_id,
            turn_id=turn_id,
            answer_clarity=AnswerClarity.CLEAR,
            navigation_values={},
            criterion_updates=[],
            next_question_id=None,
            selection_reason=(
                f"diagnostiker stub: answer to {question_id} recorded, "
                "no criterion change (the LLM arrives in Phase 3)"
            ),
        )


class VerbatimSprecher:
    """
    The verbatim Sprecher stub: it returns the released content
    UNCHANGED (mode=VERBATIM, the pieces as handed over) — the
    explicit non-default configuration of the target architecture
    (the LLM reformulation arrives in Phase 3).
    """

    def speak(
        self,
        package: ApprovedContentPackage,
        revision_instructions: str | None = None,
    ) -> SpeakerOutput:
        return SpeakerOutput(
            run_id=package.run_id,
            turn_id=package.turn_id,
            mode=SpeakerMode.VERBATIM,
            content=list(package.content),
        )


class AllowAllValidator:
    """
    The allow-all Validator stub: it releases every version (ALLOW,
    all aspects nominally checked, no reasons) — the binding exit
    control (the real checks) arrives in Phase 4, fail-closed.
    """

    def validate(
        self, output: SpeakerOutput, condition: StudyCondition | None = None
    ) -> ValidationDecision:
        return ValidationDecision(
            run_id=output.run_id,
            turn_id=output.turn_id,
            outcome=ValidationOutcome.ALLOW,
            reasons=[],
            checked_aspects=list(ValidationAspect),
            revision_instructions=None,
            canonical_fallback_used=False,
        )


class InMemoryOutputPort:
    """
    The default output port (step 2.5): it records the deliveries the
    pipeline hands over, so the terminal-mode UI and the tests can see
    what was released. Phase 5 replaces it with the real delivery
    protocol (the modality adapters stay untouched).
    """

    def __init__(self) -> None:
        self.deliveries: list[Delivery] = []

    def deliver(self, delivery: Delivery) -> None:
        self.deliveries.append(delivery)

