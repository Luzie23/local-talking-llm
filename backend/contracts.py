"""
CONTRACTS — TYPES FOR THE NEW PIPELINE'S BOUNDARY SURFACE
===========================================================
contracts & event core — types independant of logic

This module is only stdlib + Pydantic. It defines the complete boundary surface of the architecture,
so every other component (Navigator, agents, Supervisor, delivery protocol) imports from here:

  * CriterionStatus    — the four-valued criterion logic
  * ValueReading       — the Diagnostiker's current reading of one
                         navigation value (the criteria engine's input,
                         Phase 2.3)
  * QuestionState / RunStatus / DeliveryState — the state vocabularies
  * NavigationAssessment — the Diagnostiker's structured output
  * QuestionCandidate / QuestionCandidateSet — the Navigator's types
  * ApprovedContentPackage / SpeakerOutput — the Sprecher's input/output
  * ValidationDecision — the Validator's ALLOW / REVISE / BLOCK decision
  * SuicidalitySignal / AbortRequestSignal / PromptInjectionSignal — the three SEPARATE intervention signal types
    (they are never merged into one safety score)
  * StudyCondition — the study's framing/speaker configuration
  * RunState — the state snapshot the Supervisor is the sole writer of
  * the audit event schema (event classes + parse_event())
  * the ID scheme (run_id / turn_id / delivery_id)

Naming: domain-state enums (CriterionStatus, QuestionState, ...) use
UPPERCASE values because the plan and the reference scenarios speak in
those words; wire-format enums (ContentKind, EventType) use lowercase
values to stay continuous with the legacy API's "kind" strings (see
docs/legacy_api_baseline.md).

All models are frozen (immutable records; a change produces a new model)
and reject unknown fields (extra="forbid"), so that malformed agent
output fails loudly with a clear message instead of being silently
accepted.

Used by: backend/store.py and — later — every pipeline
component. The legacy pipeline does not use this module.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

# ---------------------------------------------------------------------------
# 1. State vocabularies (enums)
# ---------------------------------------------------------------------------


class CriterionStatus(str, Enum):
    """
    The four-valued criterion logic for the documentation of interview-criteria.

    OPEN and UNCLEAR are deliberately distinct:
    OPEN means the required     question has not yet been asked/answered at all
    (regular open interview work, criteria spanning over more than one question);
    UNCLEAR means an answer exists but could not be interpreted reliably
    (targeted clarification/ followup needed, detail missing in answer).
    """

    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    OPEN = "OPEN"
    UNCLEAR = "UNCLEAR"


class QuestionState(str, Enum):
    """
    Lifecycle of one question within a run.

    Clarification cycle: when the recorded answer lacks information, the
    question is REVISED and asked again (it re-enters through DELIVERED);
    the navigator must NOT move on to a new question while a question is
    REVISED. CLOSED is reached only once the answer is interpreted and no
    information is missing; SKIPPED if the question is not needed at all
    (the information is already known).

    SKIPPED: the participant already told us the information this
    question would have asked about (earlier in the conversation) — the
    question is never delivered to the participant, but the criteria that
    depend on it can still be evaluated, because the information is
    known, not missing. A skipped question is locked for this run (it is
    never asked in this run). Skipping a question does NOT end the
    interview — the run ends only via the explicit run-level transitions
    (complete_run / abort_run).

    SKIPPED also covers the PARTICIPANT-INITIATED skip (WInf-inspired,
    phase 2.5 steering): the participant explicitly asks to skip the
    CURRENT (already delivered) question ("überspringen") or refuses to
    answer it. The question then transitions DELIVERED -> SKIPPED and is
    locked just like a navigator skip — with one deliberate difference:
    the information is NOT known (nothing was recorded), so a skipped
    question NEVER counts as a negative answer: no value readings are
    taken, the criteria that depend on it stay OPEN/UNCLEAR as they were,
    and the audit cause records the skip reason (explicit / refused /
    unresolved — the per-kind re-ask budget was exhausted).
    """

    UNSEEN = "UNSEEN"                     # part of interview, not yet needed for patient
    REQUIRED = "REQUIRED"                 # needed for patient
    RESERVED = "RESERVED"                 # next to be asked via output medium
    DELIVERED = "DELIVERED"               # asked via output medium
    ANSWER_RECORDED = "ANSWER_RECORDED"   # answer recorded via input medium
    CLOSED = "CLOSED"                     # answer interpreted, no information missing
    REVISED = "REVISED"                   # answer interpreted, information missing
    SKIPPED = "SKIPPED"                   # never delivered: information already known (see class docstring)


class RunStatus(str, Enum):
    """
    Lifecycle of one interview run.

    PAUSED (WInf-inspired technical-failure policy, see SteeringConfig):
    after the configured number of CONSECUTIVE technical (model) failures
    the PARTICIPANT — not the system — may pause the run. A paused run is
    frozen (no question/criterion changes) but RESUMABLE: PAUSED -> ACTIVE
    re-opens it (the pending question is re-asked, its answer was never
    recorded). A pause is never a termination: COMPLETED/ABORTED remain
    the only terminal states.
    """

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class RunAbortCause(str, Enum):
    """
    Who/what ended a run before completion (RunAbortedEvent.cause).
    Lowercase values: this is event/wire vocabulary.
    """

    PARTICIPANT_ABORT_INTERVIEW = "participant_abort_interview"
    PARTICIPANT_ABORT_STUDY = "participant_abort_study"
    SAFETY_EARLY_STOP = "safety_early_stop"
    # the interview's routed continuation target lies OUTSIDE the loaded
    # definition (e.g. SCID A54 -> A78): an explicit, logged out-of-scope
    # exit — never a silently ignored jump.
    # Can be taken out, if interview-yaml has full interview and not only snippet.

    OUT_OF_SCOPE = "out_of_scope"


class DeliveryState(str, Enum):
    """
    Lifecycle of one delivery. The point is WHO confirms WHAT:

    RESERVED/VALIDATED exist before the participant sees anything;
    DELIVERED is logged by the OUTPUT LAYER (server side) the moment it
    has SENT the content (text payload sent / audio stream sent out) —
    it is a server-side fact and does NOT mean the participant saw or
    heard it (a client can disconnect mid-stream);
    ACKNOWLEDGED is logged when the PARTICIPANT'S CLIENT sends back an
    ACK for the delivery_id (text: content rendered; voice: audio
    actually played) — only from that moment on does the content count
    as delivered to the participant, and only then does the interview
    state advance. A missing ACK leads to a retransmission
    (delivery_retransmitted).
    """

    RESERVED = "RESERVED"
    VALIDATED = "VALIDATED"
    DELIVERED = "DELIVERED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


class AgentName(str, Enum):
    """Which LLM agent produced an agent output (audit + history keys)."""

    DIAGNOSTIKER = "diagnostiker"
    SPRECHER = "sprecher"
    VALIDATOR = "validator"


class AnswerClarity(str, Enum):
    """
    The Diagnostiker's clarity judgment on one participant answer.

    Deliberately only two values: the four-valued CriterionStatus already
    distinguishes "no answer yet" (OPEN) from "answer exists but
    unreliable" (UNCLEAR); the clarity judgment is what the Diagnostiker
    reports about a given answer.
    """

    CLEAR = "CLEAR"
    UNCLEAR = "UNCLEAR"


class ContentKind(str, Enum):
    """
    What a piece of participant-facing content is.

    Lowercase values on purpose: the legacy API already uses these strings
    for its "kind" field (docs/legacy_api_baseline.md); the two new kinds
    exist for the canonical bypass and the abort protocol.
    """

    FRAMING = "framing"
    QUESTION = "question"
    FOLLOWUP = "followup"
    FEEDBACK = "feedback"
    CLOSING = "closing"
    CANONICAL_FALLBACK = "canonical_fallback"
    ABORT_CLOSING = "abort_closing"


class ContentSource(str, Enum):
    """Where a content piece's wording comes from — a constraint the Validator can check."""

    GUIDE_VERBATIM = "guide_verbatim"
    AGENT_GENERATED = "agent_generated"
    PRE_APPROVED_FALLBACK = "pre_approved_fallback"


class SpeakerMode(str, Enum):
    """
    How the Sprecher produces its output.

    REFORMULATED is the default per the target architecture; VERBATIM is
    an explicit configuration or a technical fallback.
    """

    REFORMULATED = "REFORMULATED"
    VERBATIM = "VERBATIM"


class ValidationOutcome(str, Enum):
    """The Validator's decision. Fail-closed: any internal Validator error is a BLOCK."""

    ALLOW = "ALLOW"
    REVISE = "REVISE"
    BLOCK = "BLOCK"


class ValidationAspect(str, Enum):
    """Check aspects the Validator applies to a SpeakerOutput."""

    MEANING_FIDELITY = "MEANING_FIDELITY"    # semanitically equal to question
    TRANSPARENCY = "TRANSPARENCY"            # doesn't hinder study parameter transparency
    ROLE_CONFORMANCE = "ROLE_CONFORMANCE"    # doesn't portray role ("As you psychiatrist I want to know....")
    NON_DIAGNOSIS = "NON_DIAGNOSIS"          # doesn't lead on any symptoms or diagnosis internally confirmed for further use
    PATIENT_SAFETY = "PATIENT_SAFETY"        # no harmfull speech (most abstract criteria)


class InvariantAspect(str, Enum):
    """
    Diagnostically relevant aspects that must survive the Sprecher's
    reformulation unchanged (content, temporal reference, negation,
    answer requirement, criterion meaning).
    """

    CONTENT = "CONTENT"
    TEMPORAL_REFERENCE = "TEMPORAL_REFERENCE"
    NEGATION = "NEGATION"
    ANSWER_REQUIREMENT = "ANSWER_REQUIREMENT"
    CRITERION_MEANING = "CRITERION_MEANING"


class InterventionSignalType(str, Enum):
    """
    The three SEPARATE intervention signal classes. They must never be
    merged into one safety score, because the three classes require
    different reactions.
    """

    SUICIDALITY = "SUICIDALITY"
    ABORT_REQUEST = "ABORT_REQUEST"
    PROMPT_INJECTION = "PROMPT_INJECTION"


class DetectionLayer(str, Enum):
    """Which detection layer produced an intervention signal."""

    KEYWORD = "KEYWORD"
    LLM = "LLM"
    KEYWORD_AND_LLM = "KEYWORD_AND_LLM"


class SuicidalityAction(str, Enum):
    """
    How a fired suicidality signal is reacted to:
    HINT_ONLY  — the calm, non-blocking safety hint is shown, the
                 interview continues (the default).
    EARLY_STOP — the diagnosis is stopped early (safety protocol): the
                 run ends (run_aborted with cause=safety_early_stop); the
                 audit log is KEPT (unlike a study-wide participant abort,
                 which deletes it).
    """

    HINT_ONLY = "HINT_ONLY"
    EARLY_STOP = "EARLY_STOP"


class TransparencyMode(str, Enum):
    """
    How far the participant is informed about the AI system — the two
    study setups:
    FULL_DISCLOSURE  — the participant is exhaustively informed about
                      the system's functionality and its architecture.
    BASIC_DISCLOSURE — the participant is only informed that an AI
                      system conducts the interview, without any further
                      detail.
    The Validator's TRANSPARENCY aspect checks the SpeakerOutput against
    the configured mode (e.g. no architecture details may leak under
    BASIC_DISCLOSURE).
    """

    FULL_DISCLOSURE = "FULL_DISCLOSURE"
    BASIC_DISCLOSURE = "BASIC_DISCLOSURE"


class Modality(str, Enum):
    """
    The two study modalities a delivery reaches the participant through.
    The terminal mode is DEVELOPMENT-ONLY and not a study modality — it
    is deliberately absent here, so modality parity is defined over
    exactly text and voice.
    """

    TEXT = "text"
    VOICE = "voice"


class SubjectKind(str, Enum):
    """
    What a state-transition event is about — i.e. WHICH state machine
    changed. The four subjects have DIFFERENT state vocabularies, which
    is why from_state/to_state on the event are plain strings:
      RUN       -> RunStatus      (ACTIVE -> PAUSED / COMPLETED / ABORTED;
                                    PAUSED -> ACTIVE — the WInf-inspired
                                    technical-failure pause; only COMPLETED
                                    / ABORTED are terminal)
      QUESTION  -> QuestionState  (UNSEEN -> REQUIRED -> ... -> CLOSED;
                                   CLOSED/SKIPPED -> REQUIRED only via
                                   the episode re-open, step 2.2)
      CRITERION -> CriterionStatus (OPEN -> SATISFIED / NOT_SATISFIED /
                                     UNCLEAR)
      EPISODE   -> the episode registry of the run (declaration /
                   alias / merge of the interview's episodes — the
                   per-interview `episodes` block, design addendum
                   2026-10-26)
    subject_id names the affected question/criterion/episode
    (None for RUN).
    """

    RUN = "run"
    QUESTION = "question"
    CRITERION = "criterion"
    EPISODE = "episode"


class EventType(str, Enum):
    """Every event type the audit log can contain (lowercase: wire format)."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_ABORTED = "run_aborted"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    INPUT_RECEIVED = "input_received"
    INPUT_TRANSCRIBED = "input_transcribed"
    STATE_TRANSITION = "state_transition"
    AGENT_OUTPUT = "agent_output"
    VALIDATION_DECISION = "validation_decision"
    INTERVENTION_SIGNAL = "intervention_signal"
    DELIVERY_RESERVED = "delivery_reserved"
    DELIVERY_VALIDATED = "delivery_validated"
    DELIVERY_DELIVERED = "delivery_delivered"
    DELIVERY_ACKNOWLEDGED = "delivery_acknowledged"
    DELIVERY_RETRANSMITTED = "delivery_retransmitted"
    # (WInf-inspired) a technical (model/provider) failure of one agent
    # call — NEVER a participant action: nothing advances, nothing is
    # recorded as an answer; the participant is asked to resend, and
    # after the configured number of consecutive failures the run can
    # be PAUSED by the participant's choice.
    MODEL_FAILURE = "model_failure"


# ---------------------------------------------------------------------------
# 1b. Value readings — the criteria engine's input (Phase 2, step 2.3)
# ---------------------------------------------------------------------------


class ValueReading(BaseModel):
    """
    One navigation value's CURRENT reading in this run — the input the
    criteria rule engine (backend/criteria_engine.py) evaluates the
    rule AST over.

    Who produces it: the Diagnostiker (Phase 3), which interprets an
    answer and extracts the navigation-relevant values (plan §1.1);
    the Supervisor passes the readings to the Navigator / criteria
    engine. One reading = one value's current state.

    How the four-valued logic reads a reading (see criteria_engine.py):
      * the value's name is NOT in the readings mapping — the question
        that provides the value has not been answered yet -> OPEN
      * reliable == False — an answer exists but could not be
        interpreted reliably -> UNCLEAR
      * reliable == True — the value is evaluated; `value` of None
        means the answer carried no extractable content ("empty");
        a value that does not match the value's DECLARED type is not
        interpretable -> UNCLEAR (never silently NOT_SATISFIED)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: JsonValue = None
    reliable: bool = True


# ---------------------------------------------------------------------------
# 2. ID scheme — run_id / turn_id / delivery_id from day one
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Current timezone-aware UTC time. Every timestamp in the contracts is UTC."""
    return datetime.now(timezone.utc)


# run_<UTC compact timestamp>_<8 hex chars>
RUN_ID_PATTERN = r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}$"
# turn_001, turn_002, ... (scoped to their run, so the run id stays out of the id)
TURN_ID_PATTERN = r"^turn_\d{3,}$"
# <run_id>__d001, <run_id>__d002, ...
DELIVERY_ID_PATTERN = r"^run_\d{8}T\d{6}Z_[0-9a-f]{8}__d\d{3,}$"

RUN_ID_RE = re.compile(RUN_ID_PATTERN)
TURN_ID_RE = re.compile(TURN_ID_PATTERN)
DELIVERY_ID_RE = re.compile(DELIVERY_ID_PATTERN)


def new_run_id() -> str:
    """
    Generate a new run id (step 1.3).

    The timestamp part makes ids human-readable and sortable by start
    time; the random suffix avoids collisions between two runs started in
    the same second. This replaces the legacy fixed
    "voice_assistant_session" id: per-agent LLM chat history (if kept at
    all) is keyed by run_id, never by a fixed global session id.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z"
    return f"run_{stamp}_{secrets.token_hex(4)}"


def turn_id(sequence: int) -> str:
    """
    Build the turn id for turn number `sequence` (1-based) within a run.
    Turns are scoped to their run — the run id travels separately on
    every event — so this stays short and readable: turn_001, ....
    """
    if sequence < 1:
        raise ValueError(f"turn sequence must be >= 1, got {sequence}")
    return f"turn_{sequence:03d}"


def delivery_id(run_id: str, sequence: int) -> str:
    """
    Build a delivery id (1-based sequence within a run), derived from its
    run id so the id stays meaningful on its own — important for
    re-identification across retransmissions (Phase 5.2).
    """
    if sequence < 1:
        raise ValueError(f"delivery sequence must be >= 1, got {sequence}")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id for delivery_id: {run_id!r}")
    return f"{run_id}__d{sequence:03d}"


def history_key(run_id: str, agent: AgentName) -> str:
    """
    Key for a per-agent LLM chat history (step 1.3): if chat history is
    kept at all, it is keyed by run_id (and per agent) — never by a
    fixed global session id.
    """
    return f"{run_id}::{agent.value}"


# ---------------------------------------------------------------------------
# 3. Interventionsmonitor signals (three SEPARATE typed signals)
# ---------------------------------------------------------------------------


class _InterventionSignalBase(BaseModel):
    """Common identity of every intervention signal (internal, not a contract itself)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    detected_at: datetime = Field(default_factory=utc_now)


class SuicidalitySignal(_InterventionSignalBase):
    """
    Signal class 1: suicidality / self-harm language in the
    participant's input. Checked before regular interview processing.
    The reaction is chosen via the action field:
      HINT_ONLY (default) - the calm, non-blocking safety hint the
          frontend renders; the interview continues.
      EARLY_STOP - the diagnosis is stopped early (safety protocol):
          run_aborted(cause=safety_early_stop); the audit log is kept.
    """

    signal_type: Literal["SUICIDALITY"] = "SUICIDALITY"
    matched_keywords: list[str] = Field(default_factory=list)
    detection_layer: DetectionLayer = DetectionLayer.KEYWORD
    action: SuicidalityAction = SuicidalityAction.HINT_ONLY


class AbortRequestSignal(_InterventionSignalBase):
    """
    Signal class 2: the participant expresses an abort intent.
    Reaction (fixed): the ABORT BUTTON is highlighted — it informs the
    participant that they may abort at any time. The signal itself does
    NOT stop anything; the participant then decides via the UI:
      * abort the INTERVIEW only: run_aborted(cause=
          participant_abort_interview), optionally with free-text
          feedback (frontend text field); the audit log is kept.
      * abort the whole STUDY: run_aborted(cause=
          participant_abort_study), then the complete log (the
          participant's data) is deleted via store.abort_study() —
          only the intent to stop is recorded.
    """

    signal_type: Literal["ABORT_REQUEST"] = "ABORT_REQUEST"
    detection_layer: DetectionLayer = DetectionLayer.KEYWORD
    detected_phrase: str | None = None


class PromptInjectionSignal(_InterventionSignalBase):
    """
    Signal class 3: the participant's input looks like a prompt injection.
    An ADDITIONAL detection layer only — the system's safety guarantees
    must not depend on this signal firing.
    """

    signal_type: Literal["PROMPT_INJECTION"] = "PROMPT_INJECTION"
    detection_layer: DetectionLayer = DetectionLayer.KEYWORD
    indicators: list[str] = Field(default_factory=list)


# The three separate signal types as one discriminated union: any consumer
# receives exactly one typed signal at a time, and no type for a merged
# "safety score" exists anywhere in this module on purpose.
InterventionSignal = Annotated[
    Union[SuicidalitySignal, AbortRequestSignal, PromptInjectionSignal],
    Field(discriminator="signal_type"),
]


class InputIntent(str, Enum):
    """
    WHAT the participant is doing with one input — measured by the
    Interventionsmonitor (its second job, WInf-inspired: the monitor is
    the component that sees EVERY input before the regular processing,
    which is where intent measurement belongs — NOT the Diagnostiker,
    which only evaluates an answer against the delivered question).

    Lowercase values on purpose: wire vocabulary. The intent drives the
    deterministic steering reaction table (interview_navigator.
    steering_decision): clarification_request -> rephrase re-ask (no
    budget), cannot_answer -> one gentle follow-up then skip, refusal ->
    the skip-REQUEST reaction (the question is PROBED first — the
    participant is asked why they want to skip and offered alternatives,
    a vague/short answer is fine — and only skipped once the
    skip-probe threshold is reached), off_topic/unclear -> one redirect
    then skip; an ANSWER goes to the Diagnostiker for the regular
    clarity evaluation. The default (and the pass-through stub's) value
    is ANSWER, so the steering table degrades to the classic
    clarity-only path.
    """

    ANSWER = "answer"
    CLARIFICATION_REQUEST = "clarification_request"
    CANNOT_ANSWER = "cannot_answer"
    REFUSAL = "refusal"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"


class MonitorResult(BaseModel):
    """
    The Interventionsmonitor's result for ONE participant input
    (contracts: the monitor `check()` returns this, not a bare signal
    list): the fired intervention signals (the three SEPARATE typed
    signals, possibly empty) AND the input's intent. The signals keep
    their strict separation (never merged into one score); the intent
    is a single classification, independent of the signals (an input can
    carry a signal and any intent).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signals: tuple[InterventionSignal, ...] = ()
    intent: InputIntent = InputIntent.ANSWER


# ---------------------------------------------------------------------------
# 4. Diagnostiker output + Navigator candidate types
# ---------------------------------------------------------------------------


class CriterionUpdate(BaseModel):
    """
    One criterion-status change proposed by the Diagnostiker and applied
    by the Supervisor (the sole state writer). The Diagnostiker
    never formulates a diagnosis — only these status updates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion_id: str = Field(min_length=1)
    status: CriterionStatus
    evidence: str | None = None  # short verbatim quote from the answer backing the status
    reasoning: str | None = None  # one-sentence justification, for the audit log


class NavigationAssessment(BaseModel):
    """
    The Diagnostiker's structured output.

    The Diagnostiker evaluates an answer only as far as the steering
    decision requires: answer clarity, navigation-relevant values, and
    the choice of the contextually best next question. next_question_id
    must be one of the candidates of the QuestionCandidateSet the
    Navigator handed over — that restriction is enforced by the pipeline,
    not by this model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    answer_clarity: AnswerClarity
    # Values the rule engine (all/any/not/eq/gte/lte/threshold)
    # can consume: JSON-serializable by construction.
    navigation_values: dict[str, JsonValue] = Field(default_factory=dict)
    criterion_updates: list[CriterionUpdate] = Field(default_factory=list)
    next_question_id: str | None = None
    selection_reason: str = Field(min_length=1)


class QuestionCandidate(BaseModel):
    """One question that is currently admissible as the next question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)  # verbatim from the interview definition
    group_id: str | None = None
    reason_required: str = Field(min_length=1)  # why this question is required right now
    ordering: int = Field(default=0, ge=0)  # deterministic fallback ordering


class QuestionCandidateSet(BaseModel):
    """
    The candidate set the Navigator computes (the all_required computation)
    and the Diagnostiker must choose from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    candidates: list[QuestionCandidate] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """True if no question is currently admissible."""
        return not self.candidates

    def get(self, question_id: str) -> QuestionCandidate | None:
        """Look up one candidate by id (None if not part of the set)."""
        for candidate in self.candidates:
            if candidate.question_id == question_id:
                return candidate
        return None


# ---------------------------------------------------------------------------
# 5. Sprecher input/output + Validator decision
# ---------------------------------------------------------------------------


class ContentPiece(BaseModel):
    """One piece of participant-facing content (a question, feedback, ...)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ContentKind
    text: str = Field(min_length=1)
    source: ContentSource = ContentSource.AGENT_GENERATED


class ApprovedContentPackage(BaseModel):
    """
    The content the Diagnostiker + Navigator released, which the Sprecher
    is asked to reformulate.

    invariants lists which diagnostically relevant aspects must survive
    the reformulation unchanged — the very list the Validator checks
    against (meaning fidelity). One package per turn: it is identified
    by (run_id, turn_id).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    content: list[ContentPiece] = Field(min_length=1)
    invariants: list[InvariantAspect] = Field(
        default_factory=lambda: list(InvariantAspect)
    )
    target_language: str = "de"


class SpeakerOutput(BaseModel):
    """
    The Sprecher's output: the reformulated version of one
    ApprovedContentPackage (or the verbatim one, in explicit
    configuration/fallback mode). Only a version the Validator releases
    may reach the participant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    mode: SpeakerMode = SpeakerMode.REFORMULATED
    content: list[ContentPiece] = Field(min_length=1)


class ValidationReason(BaseModel):
    """
    One machine-readable reason attached to a ValidationDecision.
    code:   a short, STABLE identifier of the aspect/rule that fired
            (e.g. "TRANSPARENCY_VIOLATION", "MEANING_DRIFT",
            "DIAGNOSTIC_WORDING"). Stable across runs, so reasons can be
            aggregated, counted and handled programmatically.
    detail: free-text, human-readable explanation (what exactly was
            wrong) - for the audit log and the revision instructions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1)
    detail: str = ""


class ValidationDecision(BaseModel):
    """
    The Validator's binding decision on one SpeakerOutput:
    ALLOW (deliver it), REVISE (bounded revision loop — the instructions
    to revise against must be given), or BLOCK (deliver the pre-approved
    canonical fallback instead, so a session can always complete — the
    documented canonical bypass). Fail-closed: any parse or timeout
    error in the Validator itself is a BLOCK.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str = Field(pattern=TURN_ID_PATTERN)
    outcome: ValidationOutcome
    reasons: list[ValidationReason] = Field(default_factory=list)
    checked_aspects: list[ValidationAspect] = Field(
        default_factory=lambda: list(ValidationAspect)
    )
    revision_instructions: str | None = None
    canonical_fallback_used: bool = False

    @model_validator(mode="after")
    def _outcome_consistency(self) -> ValidationDecision:
        if self.outcome is ValidationOutcome.REVISE and not (
            self.revision_instructions or ""
        ).strip():
            raise ValueError(
                "a REVISE decision must carry non-empty revision_instructions "
                "(the Sprecher needs something concrete to revise against)"
            )
        if self.outcome is ValidationOutcome.BLOCK and not self.reasons:
            raise ValueError(
                "a BLOCK decision must carry at least one reason "
                "(the audit log must be able to show why the canonical "
                "fallback was used)"
            )
        if self.canonical_fallback_used and self.outcome is not ValidationOutcome.BLOCK:
            raise ValueError(
                "canonical_fallback_used is only meaningful together with outcome=BLOCK"
            )
        return self


# ---------------------------------------------------------------------------
# 6. Study condition (load this from study_settings.yaml)
# ---------------------------------------------------------------------------


class StudyCondition(BaseModel):
    """
    The study's framing configuration, supplied to both the Sprecher (as
    a constraint) and the Validator (as a check aspect).

    transparency has NO default on purpose: FULL_DISCLOSURE vs.
    BASIC_DISCLOSURE is an explicit study decision, not something to
    guess.
    speaker_enabled defaults to True because the Sprecher is part of the
    flow by default — speakerless mode is an explicit, non-default
    configuration (RECONSTRUCTION_PLAN.md §1.1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transparency: TransparencyMode
    speaker_enabled: bool = True
    interview_language: str = "de"
    behavior_profile: str | None = None


# ---------------------------------------------------------------------------
# 6b. Steering policy (load this from steering_settings.yaml)
# ---------------------------------------------------------------------------


class SteeringConfig(BaseModel):
    """
    The researcher-editable STEERING policy (WInf-inspired): how the
    deterministic reaction table (interview_navigator.steering_decision)
    treats the per-kind re-asks of a REVISED question, which input
    phrases are treated as explicit skip/continue/pause commands, how
    many consecutive technical (model) failures trigger the
    participant's pause choice, and the participant-facing fixed texts
    of that protocol.

    Loaded from study_content/steering_settings.yaml (backend/
    steering_settings.py); every field has a default here, so a missing
    file yields the defaults and a malformed value fails loudly at load
    time (extra="forbid" — no silent typos).

    The budgets are PER QUESTION and PER KIND: each re-ask kind has its
    own counter, so one redirect and one gentle follow-up can happen on
    the same question without consuming each other's budget (WInf RT-
    policy). `None` means UNLIMITED (the rephrase re-ask, WInf RT01).
    A question skipped because its budget was exhausted is recorded as
    UNRESOLVED — and never as a negative answer (see QuestionState.
    SKIPPED): no readings, the criteria stay OPEN/UNCLEAR as they were.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- the per-kind re-ask thresholds (per question) --------------------
    # the ambiguous/incomplete answer (Diagnostiker: UNCLEAR clarity)
    max_clarifications: int = Field(default=1, ge=0)
    # the off-topic / unclear answer (intents off_topic / unclear)
    max_redirects: int = Field(default=1, ge=0)
    # the "cannot answer" (intent cannot_answer)
    max_gentle_followups: int = Field(default=1, ge=0)
    # the "I didn't understand / ask again" (intent clarification_request)
    # — WInf: unlimited; None = no budget at all
    max_rephrasings: int | None = None
    # the SKIP PROBE (the explicit skip command / the refusal intent):
    # the question is NOT skipped immediately — the participant is
    # asked WHY they want to skip (and offered alternatives — a
    # vague/short answer is fine), this many times, before the question
    # is actually skipped. 0 = immediate skip (no probe).
    max_skip_probes: int = Field(default=1, ge=0)

    # -- the deterministic command phrases (exact match after
    #    normalization: lowercase, punctuation stripped, politeness
    #    fillers "bitte" / "please" ignored — WInf-inspired) ---------------
    # an input matching one of these (after normalization) skips the
    # CURRENT question — checked BEFORE the Diagnostiker (no LLM).
    skip_phrases: tuple[str, ...] = ("überspringen", "skip")
    # the failure-choice commands (after N consecutive model failures):
    continue_phrases: tuple[str, ...] = (
        "weiter",
        "weitermachen",
        "continue",
        "try again",
    )
    pause_phrases: tuple[str, ...] = (
        "pause",
        "pausieren",
        "anhalten",
        "stoppen",
    )

    # -- the technical-failure policy (WInf-inspired) ----------------------
    # a technical (model/parse) failure is NEVER a participant action:
    # nothing advances, nothing is recorded, no re-ask budget is
    # consumed; the participant is asked to resend the answer. After
    # this many CONSECUTIVE failures the participant CHOOSES: continue
    # (the counter resets) or pause the run (RunStatus.PAUSED).
    pause_after_model_failures: int = Field(default=3, ge=1)

    # -- the participant-facing fixed texts (PRE-APPROVED: they bypass
    #    the Sprecher/Validator like the terminal texts) ------------------
    # the skip probe (the participant wants to skip — explicit command /
    # refusal — and the probe budget still allows a probe): ask WHY +
    # offer the alternative (a vague/short answer is fine); it is
    # delivered with the re-asked question (FOLLOWUP):
    skip_probe_text: str = (
        "Bevor ich die Frage überspringe: Möchten Sie mir kurz sagen, "
        "warum Sie sie nicht beantworten möchten? Falls Ihnen zu viele "
        "Details schwerfallen: Sie dürfen Ihre Antwort auch vage oder "
        "kurz halten – es muss nicht ins Detail gehen."
    )
    # after one technical failure (below the threshold):
    technical_failure_text: str = (
        "Es gab ein technisches Problem: Ihre Antwort konnte nicht "
        "verarbeitet werden. Bitte senden Sie Ihre Antwort erneut."
    )
    # at the threshold: the continue / pause choice:
    failure_choice_text: str = (
        "Der technische Fehler ist mehrfach aufgetreten. Möchten Sie "
        "weitermachen (die Antwort wird erneut verarbeitet) oder die "
        "Befragung pausieren? Bitte antworten Sie mit \"weiter\" oder "
        "\"pause\"."
    )
    # the participant chose continue:
    continue_confirmation_text: str = (
        "Alles klar — bitte senden Sie Ihre Antwort erneut."
    )
    # the participant chose pause (the run is PAUSED, resumable):
    pause_confirmation_text: str = (
        "Die Befragung ist pausiert. Sie können jederzeit "
        "fortfahren — die offene Frage wird dann erneut gestellt."
    )


# ---------------------------------------------------------------------------
# 7. Run state (the Supervisor is the sole writer)
# ---------------------------------------------------------------------------


class EpisodeState(BaseModel):
    """
    One interview episode in the run's episode registry.

    The episode concept is a PER-INTERVIEW definition element (the
    `episodes` block of the definition, design addendum 2026-10-26):
    the interview first asks everything about the NOW-EPISODE (which
    must have occurred or still be occurring within its window — for
    the SCID: the last month); if that does not allow a diagnosis,
    PAST EPISODES come into play. A past episode is either PREDECLARED
    by the interview (a fixed name) or declared mid-conversation
    (dynamic, SCID style): the clinician declares "another episode"
    first, the patient then specifies it in their own words — the
    verbatim framings accumulate as `aliases`. Two declarations may
    refer to the SAME real episode ("march 2026" = "the month before I
    started uni"): when the Diagnostiker judges them identical, the
    later one is merged into the earlier (canonical) one via
    `merged_into` (one hop), so the same question is never asked twice
    for the same real episode.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["now", "past"]
    # the predeclared name (dynamic mode: the clinician's label; None
    # for the now-episode).
    label: str | None = None
    # the patient's verbatim framings ("march 2026", ...), in order.
    aliases: tuple[str, ...] = ()
    # set on the NON-canonical side of a merge (points at the winner).
    merged_into: str | None = None
    # now-episode only: the look-back window in days, plus the natural-
    # language rule carried into the Diagnostiker's prompts.
    window_days: int | None = Field(default=None, ge=1)
    rule: str | None = None


class OutOfScopeExit(BaseModel):
    """
    WHY a run ended via RunAbortCause.OUT_OF_SCOPE (design addendum
    2026-10-31, OFQ-6): the interview's deterministic routing resolved a
    jump whose continuation target lies OUTSIDE the loaded definition.
    The Supervisor aborts the run with this fact attached so the audit
    log and the frontend can render the hand-over instead of pretending
    the interview simply ended.
    Can be taken out, if interview-yaml has full interview and not only snippet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # the external target the (loaded) interview routes to — free-form
    # (it names a section/question of ANOTHER interview).
    target_ref: str = Field(min_length=1)
    # the definition's own framing of the exit (Jump.note), if any.
    note: str | None = None


class RunState(BaseModel):
    """
    The current state snapshot of one interview run.

    Frozen on purpose: changing the state means constructing a NEW
    RunState with run_version + 1 (see bumped()); the store rejects any
    version that does not advance by exactly one, which is the seed for
    the optimistic run_version check on the API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    status: RunStatus = RunStatus.ACTIVE
    definition_version: str = Field(min_length=1)
    run_version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    current_turn_id: str | None = Field(default=None, pattern=TURN_ID_PATTERN)
    question_states: dict[str, QuestionState] = Field(default_factory=dict)
    criterion_states: dict[str, CriterionStatus] = Field(default_factory=dict)
    # which episodes each question was asked about (in order; the first
    # visit is the now-episode) — the "same question never twice for
    # the same episode" invariant is checked against this.
    question_episodes: dict[str, list[str]] = Field(default_factory=dict)
    # the episode registry (id -> EpisodeState), seeded from the
    # definition's `episodes` block (always contains `now`).
    episodes: dict[str, EpisodeState] = Field(default_factory=dict)
    # the episode the CURRENT readings / criterion decisions belong to
    # ("now" until the Supervisor re-enters the interview for a past
    # episode — design addendum 2026-10-31, OFQ-1/OFQ-2).
    active_episode: str = "now"
    # set exactly once, when the run ended via RunAbortCause.OUT_OF_SCOPE
    # (the routed continuation target lies outside this definition).
    # Can be taken out, if interview-yaml has full interview and not only snippet.
    out_of_scope: OutOfScopeExit | None = None

    def bumped(self, **updates: Any) -> RunState:
        """
        Return a NEW RunState advanced by exactly one version (single-
        writer transition helper). `updates` are the state changes the
        Supervisor applied in this step.
        """
        return self.model_copy(
            update={
                "run_version": self.run_version + 1,
                "updated_at": utc_now(),
                **updates,
            }
        )


# ---------------------------------------------------------------------------
# 8. Audit event schema (one event per transition, agent
#    output, validation decision, intervention signal, delivery step)
# ---------------------------------------------------------------------------


class RunEvent(BaseModel):
    """
    Base of every audit event.

    Every event carries the definition_version of the interview definition
    the run was created from. event_id is assigned by the append-only store
    (backend/store.py): None means "not written yet", and the store
    enforces a gapless, strictly increasing sequence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: int | None = Field(default=None, ge=1)
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    turn_id: str | None = Field(default=None, pattern=TURN_ID_PATTERN)
    timestamp: datetime = Field(default_factory=utc_now)
    definition_version: str = Field(min_length=1)
    # Concrete event classes narrow this to a Literal value; the typed
    # union below (AnyRunEvent) uses it as discriminator.
    event_type: str


class RunStartedEvent(RunEvent):
    """The run was created. Carries the StudyCondition."""

    event_type: Literal["run_started"] = "run_started"
    study_condition: StudyCondition | None = None


class RunCompletedEvent(RunEvent):
    """The run finished normally (all required questions closed)."""

    event_type: Literal["run_completed"] = "run_completed"
    reason: str | None = None


class RunAbortedEvent(RunEvent):
    """
    The run ended before completion. cause names who/what ended it:
    the participant aborted the interview only (log kept, optional
    feedback from the frontend text field), the participant aborted the
    whole study (the log is deleted afterwards via
    store.RunStateRepository.abort_study - only the tombstone remains),
    or the suicidality safety protocol stopped the diagnosis early (log
    kept).
    """

    event_type: Literal["run_aborted"] = "run_aborted"
    cause: RunAbortCause
    reason: str = Field(min_length=1)
    # Optional free-text feedback the participant typed into the frontend
    # text field (interview-only abort).
    feedback: str | None = None


class TurnStartedEvent(RunEvent):
    """
    One answer-turn began. The input itself is recorded by the
    following input_received (and, for voice, input_transcribed)
    events.
    """

    event_type: Literal["turn_started"] = "turn_started"


class TurnCompletedEvent(RunEvent):
    """One answer-turn finished (delivered content acknowledged or run ended)."""

    event_type: Literal["turn_completed"] = "turn_completed"


class InputReceivedEvent(RunEvent):
    """
    The participant's input for this turn arrived (text typed in, or
    the voice recording received). The raw AUDIO itself never goes into
    the event log (binary, stays local) - only a reference to it.
    """

    event_type: Literal["input_received"] = "input_received"
    modality: Modality
    content: str | None = None      # the typed text (modality TEXT)
    audio_ref: str | None = None    # reference to the recording (modality VOICE)

    @model_validator(mode="after")
    def _modality_matches_payload(self) -> InputReceivedEvent:
        if self.modality is Modality.TEXT and not (self.content or "").strip():
            raise ValueError(
                "input_received: TEXT modality requires non-empty content (the typed text)"
            )
        if self.modality is Modality.VOICE and not (self.audio_ref or "").strip():
            raise ValueError(
                "input_received: VOICE modality requires non-empty audio_ref "
                "(reference to the recording)"
            )
        return self


class InputTranscribedEvent(RunEvent):
    """
    The STT result for a voice input. This transcript is the (normalized)
    input the Interventionsmonitor checks and the Diagnostiker evaluates.
    Voice modality only - TEXT input needs no transcription event.
    """

    event_type: Literal["input_transcribed"] = "input_transcribed"
    transcript: str = Field(min_length=1)
    language: str | None = None


class StateTransitionEvent(RunEvent):
    """
    One state change made by the Supervisor (the sole state writer).

    from_state/to_state are plain strings on purpose: the vocabulary
    depends on the subject (RunStatus vs QuestionState vs
    CriterionStatus).
    """

    event_type: Literal["state_transition"] = "state_transition"
    subject: SubjectKind
    subject_id: str | None = None  # None when subject == run
    from_state: str = Field(min_length=1)
    to_state: str = Field(min_length=1)
    cause: str = Field(min_length=1)
    # the episode the transition is about (question require/re-open:
    # which episode the question is asked about; episode registry
    # events: the affected episode). None when not episode-related.
    episode: str | None = None


class AgentOutputEvent(RunEvent):
    """
    One LLM agent output, raw + parsed. On a parse failure
    parsed is None and parse_error carries the message — the raw output
    stays in the log for debugging.
    """

    event_type: Literal["agent_output"] = "agent_output"
    agent: AgentName
    raw_output: str | None = None
    parsed: JsonValue | None = None
    parse_error: str | None = None

    @model_validator(mode="after")
    def _exactly_one_of_parsed_or_error(self) -> AgentOutputEvent:
        if self.parsed is not None and self.parse_error is not None:
            raise ValueError(
                "agent_output: parsed and parse_error are mutually exclusive "
                "(the output either parsed or it didn't)"
            )
        return self


class ValidationDecisionEvent(RunEvent):
    """The Validator's decision for the current SpeakerOutput, with its raw output."""

    event_type: Literal["validation_decision"] = "validation_decision"
    decision: ValidationDecision
    raw_output: str | None = None


class InterventionSignalEvent(RunEvent):
    """
    One fired intervention signal. The three signal classes
    remain separate here too — they are never merged into one score, and
    each has its own reaction and log entry.
    """

    event_type: Literal["intervention_signal"] = "intervention_signal"
    signal: InterventionSignal


class DeliveryReservedEvent(RunEvent):
    """Delivery lifecycle step 1: content reserved for delivery."""

    event_type: Literal["delivery_reserved"] = "delivery_reserved"
    delivery_id: str = Field(pattern=DELIVERY_ID_PATTERN)
    content_kind: ContentKind
    content_text: str = Field(min_length=1)


class DeliveryValidatedEvent(RunEvent):
    """Delivery lifecycle step 2: the Validator's outcome for this delivery is recorded."""

    event_type: Literal["delivery_validated"] = "delivery_validated"
    delivery_id: str = Field(pattern=DELIVERY_ID_PATTERN)
    outcome: ValidationOutcome


class DeliveryDeliveredEvent(RunEvent):
    """Delivery lifecycle step 3: the output layer sent the content to the participant."""

    event_type: Literal["delivery_delivered"] = "delivery_delivered"
    delivery_id: str = Field(pattern=DELIVERY_ID_PATTERN)
    modality: Modality


class DeliveryAcknowledgedEvent(RunEvent):
    """Delivery lifecycle step 4: the participant side ACKed — from now on it counts as delivered."""

    event_type: Literal["delivery_acknowledged"] = "delivery_acknowledged"
    delivery_id: str = Field(pattern=DELIVERY_ID_PATTERN)


class DeliveryRetransmittedEvent(RunEvent):
    """A delivery had to be retransmitted (missing ACK); attempts counted from 1 (Phase 5.2)."""

    event_type: Literal["delivery_retransmitted"] = "delivery_retransmitted"
    delivery_id: str = Field(pattern=DELIVERY_ID_PATTERN)
    attempts: int = Field(ge=1)


class ModelFailureEvent(RunEvent):
    """
    One TECHNICAL (model/provider) failure of one agent call
    (WInf-inspired failure policy): the agent's call or the parsing of
    its structured output failed. This is a system fact, NOT a
    participant behavior failure — the Supervisor applies the failure
    policy (no state change, no re-ask budget consumed, the participant
    is asked to resend the answer; after the configured number of
    consecutive failures the participant chooses continue/pause).
    `detail` carries the failure message for debugging/operations.
    """

    event_type: Literal["model_failure"] = "model_failure"
    agent: AgentName
    detail: str = ""


# Every event the audit log can contain, as one typed union (discriminated
# on event_type). Consumers type-hint `AnyRunEvent`; parse_event() is the
# entry point for reading events back off disk.
AnyRunEvent = Annotated[
    Union[
        RunStartedEvent,
        RunCompletedEvent,
        RunAbortedEvent,
        TurnStartedEvent,
        TurnCompletedEvent,
        InputReceivedEvent,
        InputTranscribedEvent,
        StateTransitionEvent,
        AgentOutputEvent,
        ValidationDecisionEvent,
        InterventionSignalEvent,
        DeliveryReservedEvent,
        DeliveryValidatedEvent,
        DeliveryDeliveredEvent,
        DeliveryAcknowledgedEvent,
        DeliveryRetransmittedEvent,
        ModelFailureEvent,
    ],
    Field(discriminator="event_type"),
]


# Built once at import time; the entry point for reading events back
# off disk (store.py) and for validating raw agent output in Phase 3.
_EVENT_ADAPTER: TypeAdapter[RunEvent] = TypeAdapter(AnyRunEvent)


def parse_event(data: dict[str, Any]) -> RunEvent:
    """
    Parse one JSON dict read from the event log into its concrete event
    type. Raises ValueError — with a message listing all known event
    types — if the dict is not a valid event. backend/store.py uses this
    and adds the line number to the error.
    """
    try:
        return _EVENT_ADAPTER.validate_python(data)
    except ValidationError as exc:
        known = ", ".join(sorted(et.value for et in EventType))
        raise ValueError(
            f"not a valid run event (known event types: {known}): {exc}"
        ) from exc
