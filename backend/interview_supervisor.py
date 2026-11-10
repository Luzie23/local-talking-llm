"""
INTERVIEW SUPERVISOR — THE DETERMINISTIC DRIVERS (PHASE 2, STEP 2.5)
====================================================================
The Supervisor is the SOLE writer of the InterviewRun state machine
(step 2.2) and the applier of the Navigator's proposals (step 2.4):
it requires/reserves/delivers questions, records answers, applies the
Diagnostiker's value readings (per EPISODE CONTEXT — design addendum
2026-10-31, OFQ-2), applies the online criterion decisions, and — at
GROUP COMPLETION (owner decision 2026-10-26, deferred jumps) — applies
the group consequences:

  * the skipped open questions (bypassed by a fired jump),
  * the out-of-scope exit (OFQ-6): the run is ABORTED with
    RunAbortCause.OUT_OF_SCOPE (the routed continuation target lies
    outside the loaded definition — an explicit, logged hand-over,
    never a silently swallowed jump),
  * the EPISODE re-entry (OFQ-1): a pending dynamic episode jump is
    decided — DETERMINISTICALLY by the auto policy
    (auto_episode_decisions=True, the default: the atomic
    simulation/flow) or by the LLM-driven
    apply_episode_decision() (auto_episode_decisions=False) — and the
    locked target is re-asked FOR the episode (reenter_episode:
    context switch, mood bootstrap, question re-opens, reservation),
  * the SCID A52 ESCALATION force-coding (OFQ-7): the escalation jump
    onto qg_a39 force-codes the mania duration criterion and the
    symptom threshold SATISFIED in the active context ("code A30 with
    '+' and A38 with 'JA'" — a coding decision, not a routing step).

The Supervisor owns one InterviewRun (created here) and one
InterviewNavigator; every state change goes through the state machine's
transition methods (no other mutation path) — and the Supervisor is
the only place the audit events are produced: with the default stub
wiring (below) it stays PURE and deterministic (no LLM, no I/O); the
events are collected in memory (the .events property) and, when a
store.RunStateRepository is injected, persisted in parallel to the
append-only event log and the state snapshots (run_version advances by
exactly one per saved turn — the single-writer discipline, the seed of
the optimistic run_version check of Phase 5.5).

The pipeline shape (step 2.5, fixed before any real LLM is wired in):

    Interventionsmonitor -> abort check -> Diagnostiker -> Navigator
    -> Sprecher -> Validator -> Output

(WInf-inspired) the deterministic STEERING and the technical-failure
policy (phase 2.5; the policy is researcher-editable via the
SteeringConfig, study_content/steering_settings.yaml):
  * the Interventionsmonitor measures the input's INTENT (contracts.
    InputIntent) in addition to the three signal classes;
  * BEFORE the Diagnostiker, two LLM-free pre-checks handle the SKIP
    REQUESTS — the explicit skip command (normalized exact match) and
    the refusal intent: the CURRENT question is NOT skipped
    immediately, the system ASKS the participant why they want to
    skip (and offers alternatives: a vague/short answer is fine — the
    pre-approved skip_probe_text, re-asked as a FOLLOWUP with its own
    skip-probe counter); only once the skip-probe threshold
    (max_skip_probes) is reached the question is skipped
    (DELIVERED -> SKIPPED);
  * AFTER the Diagnostiker, the deterministic reaction table
    (interview_navigator.steering_decision — intent + clarity + the
    per-kind re-ask counters + the configured thresholds) decides
    PROCEED (close) / REASK (the question is REVISED and re-asked as
    clarify / redirect / gentle / rephrase / skip-probe, each kind
    with its own counter) / SKIP (budget exhausted — unresolved);
  * a skipped question is NEVER a negative answer: no readings, the
    criteria depending on it stay OPEN/UNCLEAR as they were;
  * a technical (model) failure of an agent call is NEVER a
    participant action: no state change, no budget consumed, the
    model_failure audit event is logged and the participant is asked
    to resend; after the configured number of consecutive failures
    the participant CHOOSES continue (counter resets) or pause the
    run (RunStatus.PAUSED — frozen, resumable via resume_run(), the
    pending question is re-asked).

The four agent components are pluggable behind the protocols of
backend/interview_agents.py; the DEFAULT wiring is the deterministic
stubs (the pass-through Monitor; the "answer recorded, no criterion
change" Diagnostiker stub; the verbatim Sprecher stub; the allow-all
Validator stub) and the InMemoryOutputPort (Phase 5 wires the real
delivery protocol — reservation, ACK, retransmission — the modality
adapters stay untouched).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Mapping, Sequence

from contracts import (
    AbortRequestSignal,
    AgentName,
    AgentOutputEvent,
    ApprovedContentPackage,
    ContentKind,
    ContentPiece,
    ContentSource,
    CriterionStatus,
    DeliveryDeliveredEvent,
    DeliveryReservedEvent,
    DeliveryValidatedEvent,
    InvariantAspect,
    InputIntent,
    InputReceivedEvent,
    InputTranscribedEvent,
    InterventionSignal,
    InterventionSignalEvent,
    Modality,
    ModelFailureEvent,
    OutOfScopeExit,
    QuestionState,
    RunAbortedEvent,
    RunAbortCause,
    RunCompletedEvent,
    RunEvent,
    RunStartedEvent,
    RunStatus,
    SpeakerOutput,
    SteeringConfig,
    StudyCondition,
    SuicidalityAction,
    SuicidalitySignal,
    TurnCompletedEvent,
    TurnStartedEvent,
    ValidationAspect,
    ValidationDecision,
    ValidationDecisionEvent,
    ValidationOutcome,
    ValidationReason,
    ValueReading,
    delivery_id,
    new_run_id,
    turn_id as make_turn_id,
)
from interview_agents import (
    AllowAllValidator,
    Delivery,
    Diagnostiker,
    InMemoryOutputPort,
    InterventionsMonitor,
    OutputPort,
    PassthroughMonitor,
    Sprecher,
    StubDiagnostiker,
    Validator,
    VerbatimSprecher,
)
from interview_definition import (
    ForcedCriterion,
    InterviewDefinition,
    QuestionGroup,
)
from interview_navigator import (
    EpisodeDecision,
    InterviewNavigator,
    JumpResolution,
    LoopPolicy,
    NavigatorError,
    QuestionCandidateSet,
    ReaskKind,
    SkipReason,
    SteeringAction,
    is_explicit_skip,
    normalize_command,
    skip_request_decision,
    steering_decision,
)
from interview_run import IllegalTransitionError, InterviewRun
from store import EventStore, RunStateRepository


def _is_true_reading(reading: ValueReading | None) -> bool:
    """A reliable reading whose value is exactly True."""
    return reading is not None and reading.reliable and reading.value is True


@dataclass(frozen=True)
class SupervisorOutcome:
    """
    The outcome of one Supervisor step. `kind` is the flow state after
    the step:
      * continue          — the interview continues; next_question is
                            the reserved next question;
      * awaiting_answer   — the question was delivered; the
                            participant's answer is awaited;
      * answer_in_flight  — nothing can be asked (an answer is in
                            flight); no reservation was made;
      * pending_episode   — a dynamic episode jump awaits the
                            Diagnostiker's EpisodeDecision
                            (apply_episode_decision; the LLM-driven
                            mode only — the auto mode decides inline);
      * out_of_scope      — the run was ABORTED via the out-of-scope
                            exit (OFQ-6): the continuation target lies
                            outside this definition (message carries
                            the target/note);
      * complete          — the run was COMPLETED (nothing more to ask).
    """

    CONTINUE: ClassVar[str] = "continue"
    AWAITING_ANSWER: ClassVar[str] = "awaiting_answer"
    ANSWER_IN_FLIGHT: ClassVar[str] = "answer_in_flight"
    PENDING_EPISODE: ClassVar[str] = "pending_episode"
    OUT_OF_SCOPE: ClassVar[str] = "out_of_scope"
    COMPLETE: ClassVar[str] = "complete"

    kind: str
    next_question: str | None = None
    episode: str | None = None
    message: str = ""
    pending_source: str | None = None


@dataclass(frozen=True)
class TurnOutcome:
    """
    The outcome of one PIPELINE turn (start / receive_input / the
    participant aborts / the episode decision) — as opposed to
    SupervisorOutcome, which describes one deterministic step.
    `delivered` is the RELEASED content piece the output layer hands
    to the participant (None when nothing was delivered — a study
    abort, a pending episode); `signals` are the fired intervention
    signals as the three SEPARATE typed signals (never merged);
    `abort_available` tells the UI to highlight the abort button
    (an AbortRequestSignal fired — the interview itself continues,
    the participant decides via the UI); `out_of_scope` is set when
    the turn ended the run via the out-of-scope exit (OFQ-6);
    `pending_episode` is set when a dynamic episode jump awaits the
    Diagnostiker's EpisodeDecision (complete_pending_episode);
    (WInf-inspired technical-failure policy) `model_failure` is set
    when the turn was the REACTION to a technical (model) failure
    (nothing advanced — the participant is asked to resend the
    answer), `consecutive_model_failures` carries the current count
    (it restarts at a successful turn) and `failure_choice_pending`
    is set when the threshold was reached and the participant must
    now choose: continue (the pending answer is retried) or pause the
    run (run_status becomes PAUSED, resumable via resume_run()).
    """

    turn_id: str | None
    run_status: RunStatus
    delivered: ContentPiece | None = None
    delivery_id: str | None = None
    question_id: str | None = None
    signals: tuple[InterventionSignal, ...] = ()
    abort_available: bool = False
    out_of_scope: OutOfScopeExit | None = None
    pending_episode: bool = False
    # (WInf-inspired) the technical-failure policy:
    model_failure: bool = False
    consecutive_model_failures: int = 0
    failure_choice_pending: bool = False
    reason: str = ""


# The participant-facing default texts of the pipeline's TERMINAL
# steps (step 2.5): the run's CLOSING (the interview is complete), the
# ABORT CLOSING (the participant aborted the interview) and the SAFETY
# EARLY-STOP hint. They are PRE-APPROVED (ContentSource.
# PRE_APPROVED_FALLBACK): like the canonical fallback, they bypass the
# Speaker and the Validator by design (the binding exit control applies
# to the Speaker's versions). Researcher-editable study content in the
# real deployment (see study_content/); here: German defaults.
DEFAULT_CLOSING_TEXT = (
    "Die Befragung ist damit abgeschlossen. Vielen Dank für Ihre "
    "Teilnahme."
)
DEFAULT_ABORT_CLOSING_TEXT = (
    "Der Interviewverlauf wurde beendet. Sie können die Teilnahme "
    "jederzeit beenden."
)
DEFAULT_SAFETY_HINT_TEXT = (
    "Sie beschreiben gerade schwer belastende Gedanken. Hier finden "
    "Sie Unterstützung: [Krisenkontakte einfügen]. Der Interviewverlauf "
    "wird daraufhin beendet."
)
# the OUT-OF-SCOPE hand-over (OFQ-6): the routed continuation target
# lies outside this definition — {target} is the external target the
# interview routes to (the definition's own note, if any, is appended).
DEFAULT_OUT_OF_SCOPE_TEMPLATE = (
    "Dieser Abschnitt ist hier zu Ende: der weitere Verlauf führt "
    "weiter mit {target}."
)
# The audit-cause wording for the run's COMPLETION (step 2.5). Unlike the
# DEFAULT_* texts above (participant-facing), this is an AUDIT string —
# written into the transition log (the complete_run cause) and the
# run_completed event reason. A single source of truth so the cause and
# the event reason agree. (The loop-policy completion is a DISTINCT
# cause, kept parameterized at its call site.)
CAUSE_RUN_COMPLETE_NO_QUESTIONS = "no more questions — the interview is complete"


class InterviewSupervisor:
    """
    The deterministic driver of ONE interview run (see the module
    docstring). The Supervisor creates and owns the run
    (InterviewSupervisor(definition) — or with an explicit run_id) and
    exposes the question flow (require_and_reserve_next / deliver /
    apply_answer / candidate_set) plus the episode re-entry API
    (reenter_episode / apply_episode_decision).
    """

    def __init__(
        self,
        definition: InterviewDefinition,
        run_id: str | None = None,
        loop_policy: LoopPolicy = LoopPolicy.IGNORE_AND_CONTINUE,
        auto_episode_decisions: bool = True,
        max_past_episodes: int = 5,
        *,
        study_condition: StudyCondition | None = None,
        monitor: InterventionsMonitor | None = None,
        diagnostiker: Diagnostiker | None = None,
        speaker: Sprecher | None = None,
        validator: Validator | None = None,
        output_port: OutputPort | None = None,
        repository: RunStateRepository | None = None,
        modality: Modality = Modality.TEXT,
        closing_text: str = DEFAULT_CLOSING_TEXT,
        abort_closing_text: str = DEFAULT_ABORT_CLOSING_TEXT,
        safety_hint_text: str = DEFAULT_SAFETY_HINT_TEXT,
        out_of_scope_template: str = DEFAULT_OUT_OF_SCOPE_TEMPLATE,
        max_validation_rounds: int = 2,
        steering: SteeringConfig | None = None,
    ) -> None:
        self._definition = definition
        self._run = InterviewRun(definition, run_id or new_run_id())
        self._navigator = InterviewNavigator(definition, loop_policy)
        self._auto_episode_decisions = auto_episode_decisions
        self._max_past_episodes = max(1, max_past_episodes)
        # (WInf-inspired) the steering policy: the per-kind re-ask
        # thresholds, the explicit skip / continue / pause command
        # phrases, the technical-failure pause threshold and the
        # fixed texts of the failure/pause protocol. Default: the
        # built-in SteeringConfig (pure, no I/O — the deployment wiring
        # passes load_steering_settings() from backend/
        # steering_settings.py).
        self._steering = steering if steering is not None else SteeringConfig()
        # (group_id, episode context) pairs whose consequences were
        # already applied (idempotent group completion across
        # re-entries: a group re-worked for another episode is
        # completed AGAIN for that context — a distinct pair).
        self._completed_groups: set[tuple[str, str]] = set()
        self._past_episodes_declared = 0
        self._group_of: dict[str, QuestionGroup] = {
            q.id: group for group, q in definition.questions_with_group()
        }
        # the pipeline wiring (step 2.5): the study condition, the
        # agent components (default: the deterministic stubs of
        # backend/interview_agents.py), the output port, the session
        # modality and the terminal texts.
        self._study_condition = study_condition
        self._monitor = monitor if monitor is not None else PassthroughMonitor()
        self._diagnostiker = (
            diagnostiker if diagnostiker is not None else StubDiagnostiker()
        )
        self._sprecher = speaker if speaker is not None else VerbatimSprecher()
        self._validator = validator if validator is not None else AllowAllValidator()
        self._output_port: OutputPort = (
            output_port if output_port is not None else InMemoryOutputPort()
        )
        self._modality = modality
        self._closing_text = closing_text
        self._abort_closing_text = abort_closing_text
        self._safety_hint_text = safety_hint_text
        self._out_of_scope_template = out_of_scope_template
        self._max_validation_rounds = max(1, max_validation_rounds)
        # the audit trail: every state change, agent output, validation
        # decision, intervention signal, input and delivery step is
        # logged HERE (the single writer) — in memory (the .events
        # property) and, when a repository was given, in parallel to
        # the append-only event log; the state snapshots are saved once
        # per turn (run_version +1 — the store rejects anything else).
        self._events: list[RunEvent] = []
        self._emitted_transitions = 0
        self._repo: RunStateRepository | None = repository
        self._events_store: EventStore | None = None
        self._run_version = 1
        self._study_aborted = False
        # turn / delivery bookkeeping (the id scheme of contracts.py):
        self._turn_sequence = 0
        self._current_turn_id: str | None = None
        self._delivery_sequence = 0
        self._started = False
        # the source question of a PENDING dynamic episode jump (the
        # LLM-driven mode; set when a PENDING_EPISODE outcome is
        # returned, consumed by complete_pending_episode).
        self._pending_source: str | None = None
        # (WInf-inspired) the steering / failure-policy bookkeeping —
        # operational state of the PIPELINE (not of the run's state
        # machine): the consecutive technical (model) failure count
        # (restarts at a successful turn / a 'continue' choice / a
        # resume), the pending continue/pause choice after the failure
        # threshold, and the per-kind re-ask counters of the in-flight
        # question (question_id -> {kind wire value: count} — the
        # per-kind attempt counters; reset when the question leaves
        # the in-flight state or is re-asked for another episode).
        # The AUDIT TRAIL of both policies lives in the event stream
        # (the model_failure events, the skip/revise causes, the
        # state transitions) — see RECONSTRUCTION_PLAN.md §5 (open
        # question 7) on persisting the counters in the run state.
        self._consecutive_model_failures = 0
        self._failure_choice_pending = False
        self._reask_counts: dict[str, dict[str, int]] = {}
        if repository is not None:
            repository.create_run(self._run.snapshot(1))
            self._events_store = repository.event_store(self.run_id)
        self._emit_run_started()

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def definition(self) -> InterviewDefinition:
        return self._definition

    @property
    def run(self) -> InterviewRun:
        return self._run

    @property
    def navigator(self) -> InterviewNavigator:
        return self._navigator

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def status(self) -> RunStatus:
        return self._run.status

    @property
    def active_episode(self) -> str:
        return self._run.active_episode

    @property
    def out_of_scope(self):
        """The out-of-scope exit (set once, when the run aborted via OFQ-6)."""
        return self._run.out_of_scope

    @property
    def episodes(self) -> dict:
        return self._run.episodes

    def question_state(self, question_id: str) -> QuestionState:
        return self._run.question_state(question_id)

    @property
    def question_states(self) -> dict[str, QuestionState]:
        """A copy of ALL question states (question_id -> state)."""
        return self._run.question_states

    def criterion_state(self, criterion_id: str) -> CriterionStatus:
        return self._run.criterion_state(criterion_id)

    def readings(
        self, episode_id: str | None = None
    ) -> dict[str, ValueReading]:
        """A copy of one episode context's readings (default: active)."""
        return self._run.readings(episode_id)

    def value(
        self, name: str, episode_id: str | None = None
    ) -> ValueReading | None:
        """The current reading of one value (default: the active context)."""
        return self._run.value(name, episode_id)

    def readings_by_context(self) -> dict[str, dict[str, ValueReading]]:
        """A copy of ALL per-context readings (context -> readings)."""
        return self._run.readings_by_context()

    def next_question_text(self) -> str | None:
        """
        The canonical text (with the {text tokens} rendered from the
        run's current state — OFQ-4/OFQ-5) of the next question, or
        None when there is nothing to ask right now.
        """
        question_id = self._navigator.next_question(self._run)
        if question_id is None:
            return None
        return self._navigator.rendered_text(question_id, self._run)

    @property
    def events(self) -> tuple[RunEvent, ...]:
        """
        The run's full audit event stream (in order): run_started,
        every state transition (via TransitionRecord.to_event()),
        every agent output, validation decision, intervention signal,
        input, turn and delivery step. The in-memory stream is the
        canonical one; when a repository was configured, the very
        same events were appended to the append-only event log in
        parallel (store.py).
        """
        return tuple(self._events)

    @property
    def run_version(self) -> int:
        """
        The run_version of the LAST SAVED state snapshot: 1 when no
        turn has been saved yet (create_run wrote version 1), then
        exactly one more per saved turn — the seed of the optimistic
        run_version check (Phase 5.5).
        """
        return self._run_version

    @property
    def output_port(self) -> OutputPort:
        """The output layer's port (the released deliveries can be
        read back from the default InMemoryOutputPort)."""
        return self._output_port

    @property
    def study_condition(self) -> StudyCondition | None:
        return self._study_condition

    @property
    def pending_source(self) -> str | None:
        """
        The source question of the PENDING dynamic episode jump (the
        LLM-driven mode), if one awaits the Diagnostiker's
        EpisodeDecision (complete_pending_episode); None otherwise.
        """
        return self._pending_source

    @property
    def steering(self) -> SteeringConfig:
        """The steering policy applied by the pipeline (WInf-inspired)."""
        return self._steering

    @property
    def consecutive_model_failures(self) -> int:
        """(WInf-inspired) the current consecutive technical-failure count."""
        return self._consecutive_model_failures

    @property
    def failure_choice_pending(self) -> bool:
        """
        (WInf-inspired) True while the participant must still answer the
        continue/pause choice offered after the technical-failure
        threshold (the next input is the deterministic choice check).
        """
        return self._failure_choice_pending


    # ------------------------------------------------------------------
    # the question flow (require / reserve / deliver / answer)
    # ------------------------------------------------------------------

    def candidate_set(self, turn_id: str) -> QuestionCandidateSet:
        """The candidate set handed to the interviewer agent (step 2.4)."""
        return self._navigator.candidate_set(self._run, turn_id)

    def require_and_reserve_next(
        self, cause: str = "supervisor: next question"
    ) -> SupervisorOutcome:
        """
        Advance to the next question: require + reserve the
        deterministic fallback next question (or the re-asked REVISED
        one — which re-enters directly via DELIVERED, so it is NOT
        reserved: REVISED -> RESERVED is not a legal transition, the
        state machine's re-ask move is REVISED -> DELIVERED).
        Completes the run when nothing more is to be asked;
        reports answer_in_flight when an answer is awaited.
        """
        if self._require_active():
            return SupervisorOutcome(kind=SupervisorOutcome.ANSWER_IN_FLIGHT)
        question_id = self._navigator.next_question(self._run)
        if question_id is None:
            if self._navigator.is_complete(self._run):
                self._run.complete_run(
                    f"supervisor: {CAUSE_RUN_COMPLETE_NO_QUESTIONS}"
                )
                return SupervisorOutcome(kind=SupervisorOutcome.COMPLETE)
            return SupervisorOutcome(kind=SupervisorOutcome.ANSWER_IN_FLIGHT)
        if self._run.question_state(question_id) is QuestionState.REVISED:
            # the re-asked REVISED question re-enters directly via
            # DELIVERED (deliver()) — no reservation on the way.
            return SupervisorOutcome(
                kind=SupervisorOutcome.CONTINUE, next_question=question_id
            )
        self._ensure_required(
            question_id,
            f"{cause} (episode {self._run.active_episode})",
        )
        if self._run.question_state(question_id) is not QuestionState.RESERVED:
            self._run.reserve_question(question_id, cause)
        return SupervisorOutcome(
            kind=SupervisorOutcome.CONTINUE, next_question=question_id
        )

    def deliver(
        self, cause: str = "output layer: question sent"
    ) -> SupervisorOutcome:
        """
        Deliver the reserved question via the output layer — or the
        re-asked REVISED question, which re-enters directly via
        DELIVERED (no reservation on the way; the single-writer move
        is the same deliver_question transition either way).
        """
        self._require_active()
        question_id = self._run.reserved_question()
        if question_id is None:
            revised = self._run.revised_questions()
            if len(revised) > 1:
                raise NavigatorError(
                    "multiple REVISED questions — exactly one may be "
                    "re-asked at a time"
                )
            if not revised:
                raise NavigatorError(
                    "no question is reserved — require_and_reserve_next() "
                    "first"
                )
            question_id = revised[0]
        self._run.deliver_question(question_id, cause)
        return SupervisorOutcome(
            kind=SupervisorOutcome.AWAITING_ANSWER, next_question=question_id
        )

    def apply_answer(
        self,
        question_id: str,
        readings: Mapping[str, ValueReading] | Sequence[ValueReading] = (),
        closed: bool = True,
        cause: str = "interpreted answer",
    ) -> SupervisorOutcome:
        """
        Apply the Diagnostiker's interpretation of ONE answer (the
        atomic flow's main step): record the answer (when still
        DELIVERED), store the value readings in the ACTIVE episode
        context (OFQ-2), evaluate the criteria online (and apply the
        proposals), and — once the question's group is COMPLETE —
        resolve the group's DEFERRED jump (skips, the out-of-scope
        exit, the episode re-entry, the escalation force-coding) and
        reserve the next question.
        """
        self._require_active()
        state = self._run.question_state(question_id)
        if state is QuestionState.DELIVERED:
            self._run.record_answer(question_id, cause)
            state = self._run.question_state(question_id)
        if state is not QuestionState.ANSWER_RECORDED:
            raise NavigatorError(
                f"question {question_id!r} is {state.value} — an answer "
                f"can only be applied to a DELIVERED/ANSWER_RECORDED "
                f"question"
            )
        readings_map = self._normalize_readings(readings)
        for reading in readings_map.values():
            self._run.set_value(reading, cause)
        if closed:
            self._run.close_question(question_id, cause)
        # online criteria (the active episode context — OFQ-2)
        proposals = self._navigator.propose_criterion_updates(
            self._run, readings_map
        )
        for criterion_id, status in proposals.items():
            self._run.set_criterion(
                criterion_id,
                status,
                cause=f"criteria engine: {criterion_id} after {question_id}",
            )
        # the DEFERRED jump: resolved once the question's group is
        # complete (owner decision 2026-10-26) — applied ONCE per
        # (group, episode context).
        group = self._group_of[question_id]
        context = self._run.active_episode
        if (
            self._navigator.group_is_complete(self._run, group.id)
            and (group.id, context) not in self._completed_groups
        ):
            self._completed_groups.add((group.id, context))
            return self._apply_group_completion(question_id, group, cause)
        return SupervisorOutcome(
            kind=SupervisorOutcome.CONTINUE,
            next_question=self._navigator.next_question(self._run),
        )


    # ------------------------------------------------------------------
    # the group consequences (deferred jumps, applied ONCE per context)
    # ------------------------------------------------------------------

    def _apply_group_completion(
        self, source_id: str, group: QuestionGroup, cause: str
    ) -> SupervisorOutcome:
        completion = self._navigator.complete_group(self._run, group.id)
        for qid in completion.optional_skip_proposals:
            self._skip(
                qid,
                f"supervisor: optional question not needed once group "
                f"{group.id} is complete",
            )
        resolution = completion.resolution
        if resolution.out_of_scope:
            return self._finish_out_of_scope(source_id, resolution)
        if resolution.pending_episode:
            if not self._auto_episode_decisions:
                return SupervisorOutcome(
                    kind=SupervisorOutcome.PENDING_EPISODE,
                    pending_source=source_id,
                    message=resolution.loop_note or "",
                )
            decision, _episode = self._decide_episode(
                resolution.jump, source_id
            )
            resolution = self._navigator.resolve_jump(
                self._run, source_id, episode_decision=decision
            )
        return self._finish_resolution(source_id, resolution)

    def _finish_resolution(
        self, source_id: str, resolution: JumpResolution
    ) -> SupervisorOutcome:
        """Apply a FINAL resolution: skips, re-entry, escalation, next."""
        if resolution.out_of_scope:
            return self._finish_out_of_scope(source_id, resolution)
        for qid in resolution.skip_proposals:
            self._skip(
                qid,
                f"supervisor: bypassed by the fired jump of {source_id} "
                f"(not needed in the current flow)",
            )
        if resolution.reopen_proposals:
            # the RE-ENTRY (OFQ-1): re-ask the locked target FOR the
            # episode (context switch, mood bootstrap, re-opens,
            # reservation).
            episode = resolution.episode
            target = resolution.target_id
            if episode is None or target is None:
                raise NavigatorError(
                    "re-entry proposals without an episode and target "
                    "— the resolution is inconsistent"
                )
            self.reenter_episode(
                episode,
                target,
                cause=(
                    f"supervisor: episode jump of {source_id} re-enters "
                    f"the interview"
                ),
            )
        # the DECLARATIVE force-coding (OFQ-7): a jump that declares
        # `force_criterion` and APPLIED (target re-asked or first-asked
        # — not a locked target that was ignored) force-codes its
        # declared criteria in the active context. The SCID A52
        # escalation declares this on its jump onto qg_a39: "code A30
        # with '+' and A38 with 'JA'".
        if (
            resolution.fired
            and resolution.jump is not None
            and resolution.jump.force_criterion
            and not resolution.locked_target
        ):
            self._apply_jump_force_coding(
                source_id, resolution.jump.force_criterion
            )
        if resolution.policy_applied is LoopPolicy.END_RUN:
            self._run.complete_run(
                f"supervisor: the loop policy ends the run after {source_id}"
            )
            return SupervisorOutcome(kind=SupervisorOutcome.COMPLETE)
        # the LIVE next question (the skips just applied may have
        # removed the resolution's snapshot next_question — e.g. the
        # group's optional questions at completion):
        next_question = self._navigator.next_question(self._run)
        if next_question is not None:
            self._ensure_required(
                next_question,
                f"supervisor: next question after {source_id} "
                f"(episode {self._run.active_episode})",
            )
            # the re-entry already reserved the first re-opened
            # question — reserve everything else:
            if (
                self._run.question_state(next_question)
                is not QuestionState.RESERVED
            ):
                self._run.reserve_question(
                    next_question, f"supervisor: reserving {next_question}"
                )
        return SupervisorOutcome(
            kind=SupervisorOutcome.CONTINUE,
            next_question=next_question,
            episode=resolution.episode,
        )

    def _finish_out_of_scope(
        self, source_id: str, resolution: JumpResolution
    ) -> SupervisorOutcome:
        """The OUT-OF-SCOPE exit (OFQ-6): an explicit, logged abort."""
        jump = resolution.jump
        target = resolution.target_id
        if jump is None or target is None:
            raise NavigatorError(
                "an out-of-scope resolution without a jump and target "
                "— the resolution is inconsistent"
            )
        self._run.abort_out_of_scope(
            target,
            jump.note or "",
            cause=(
                f"supervisor: out-of-scope exit after {source_id} — the "
                f"interview continues with {target} "
                f"(outside this definition)"
            ),
        )
        return SupervisorOutcome(
            kind=SupervisorOutcome.OUT_OF_SCOPE,
            message=(
                f"out-of-scope exit after {source_id}: the interview "
                f"continues with {target} "
                f"({jump.note or 'outside this definition'})"
            ),
        )


    # ------------------------------------------------------------------
    # the episode re-entry (OFQ-1)
    # ------------------------------------------------------------------

    def reenter_episode(
        self,
        episode_id: str,
        target_group: str,
        cause: str = "supervisor: episode re-entry",
    ) -> SupervisorOutcome:
        """
        The RE-ENTRY API (OFQ-1): open `target_group` (a locked jump
        target) FOR `episode_id` — the interview re-enters the locked
        span for the new episode:
          1. switch the ACTIVE episode context (set_active_episode);
          2. bootstrap the episode's mood context (the A54 screen of
             the previous context pinned past_* -> episode_*);
          3. re-open every terminal question of the target group for
             the episode (reopen_question — the state machine's only
             move out of a terminal question state);
          4. reserve the first re-opened question (the deterministic
             next question).
        The episode must exist in the registry (declare it first in
        dynamic mode — the Supervisor's auto policy does this).
        """
        self._require_active()
        previous_context = self._run.active_episode
        self._run.set_active_episode(episode_id, cause)
        self._bootstrap_mood(previous_context, episode_id)
        target_questions = self._questions_of(target_group)
        reopened: list[str] = []
        for qid in target_questions:
            state = self._run.question_state(qid)
            if state in (QuestionState.CLOSED, QuestionState.SKIPPED):
                self._run.reopen_question(qid, episode_id, cause)
                reopened.append(qid)
        if not reopened:
            raise NavigatorError(
                f"episode re-entry failed: target {target_group!r} has "
                f"no re-askable (terminal) questions for episode "
                f"{episode_id!r}"
            )
        # (WInf-inspired) a new episode is a NEW round: the per-kind
        # re-ask counters of the re-opened questions start fresh.
        for qid in reopened:
            self._reask_counts.pop(qid, None)
        first = reopened[0]
        self._run.reserve_question(
            first, f"{cause} (first re-opened question {first})"
        )
        return SupervisorOutcome(
            kind=SupervisorOutcome.CONTINUE,
            next_question=first,
            episode=episode_id,
        )

    def apply_episode_decision(
        self, question_id: str, decision: EpisodeDecision
    ) -> SupervisorOutcome:
        """
        The LLM-driven episode decision (OFQ-1): apply the
        Diagnostiker's EpisodeDecision to a PENDING dynamic episode
        jump declared on `question_id` (the LLM-driven mode,
        auto_episode_decisions=False) and finish applying the
        resolution (re-entry, escalation force-coding, next question).
        """
        self._require_active()
        resolution = self._navigator.resolve_jump(
            self._run, question_id, episode_decision=decision
        )
        if resolution.out_of_scope:
            return self._finish_out_of_scope(question_id, resolution)
        return self._finish_resolution(question_id, resolution)


    def _decide_episode(
        self, jump, source_id: str
    ) -> tuple[EpisodeDecision, str]:
        """
        The DETERMINISTIC auto policy for a PENDING dynamic episode
        jump (OFQ-1, the atomic mode):
          * the target is NOT yet assessed for the ACTIVE episode ->
            the continuation re-runs the A54-confirmed battery FOR the
            active episode (it was re-entered before; e.g. ep_1 after
            the A54 screen confirmed a past episode);
          * the target IS already assessed for the active episode (the
            A54 confirmation loop) -> a NEW past episode is declared
            and re-entered (bounded by max_past_episodes).
        """
        active = self._run.active_episode
        if not self._navigator.target_assessed_for(self._run, jump.to, active):
            return (
                EpisodeDecision(active, EpisodeDecision.NEW_EPISODE),
                active,
            )
        if self._past_episodes_declared >= self._max_past_episodes:
            raise NavigatorError(
                f"episode decision: the maximum number of past episodes "
                f"({self._max_past_episodes}) was reached — the "
                f"interview cannot re-enter further episodes"
            )
        record = self._run.declare_episode(
            f"frühere Episode {self._past_episodes_declared + 1}",
            f"supervisor: {source_id} confirms another past episode",
        )
        self._past_episodes_declared += 1
        return (
            EpisodeDecision(record.subject_id, EpisodeDecision.NEW_EPISODE),
            record.subject_id,
        )

    def _bootstrap_mood(self, previous_context: str, episode: str) -> None:
        """
        Seed the re-entered episode's mood context (OFQ-1): the A54
        screen answered in `previous_context` pinned the PAST episode's
        mood (past_elevated_mood / past_irritable_mood); the re-entered
        battery needs the episode's mood values (episode_elevated_mood
        / episode_irritable_mood) to evaluate its criteria and render
        the {symptom_term} token.
        """
        cause = "supervisor: re-entry mood bootstrap (the A54 screen)"
        source = self._run.readings(previous_context)
        if _is_true_reading(source.get("past_elevated_mood")):
            self._run.set_value_for(
                episode,
                ValueReading(name="episode_elevated_mood", value=True),
                cause,
            )
        if _is_true_reading(source.get("past_irritable_mood")):
            self._run.set_value_for(
                episode,
                ValueReading(name="episode_irritable_mood", value=True),
                cause,
            )

    def _apply_jump_force_coding(
        self, source_id: str, forced: list[ForcedCriterion]
    ) -> None:
        """
        Apply a jump's DECLARED force-coding (design addendum 2026-11-03,
        OFQ-7): each entry the jump's `force_criterion` lists is
        force-coded to its status in the ACTIVE episode context — a
        coding decision, not a routing step (the transition table is
        respected — a decided criterion is reopened via UNCLEAR,
        run.force_criterion). The SCID A52 escalation declares its
        "kodieren Sie A30 mit '+' und A38 mit 'JA'" on the jump itself.
        """
        for entry in forced:
            if self._definition.criterion_by_id(entry.criterion) is not None:
                self._run.force_criterion(
                    entry.criterion,
                    entry.status,
                    f"supervisor: jump of {source_id} force-codes "
                    f"{entry.criterion} -> {entry.status.value}",
                )


    # ------------------------------------------------------------------
    # the pipeline (step 2.5): Monitor -> abort check -> Diagnostiker
    # -> Navigator -> Sprecher -> Validator -> Output
    # ------------------------------------------------------------------

    def start(self) -> TurnOutcome:
        """
        The OPENING turn (turn_001): no participant input — the
        Navigator's first question is released through the
        Sprecher -> Validator -> Output chain and handed to the output
        layer (the delivery events + the run_started event are logged).
        """
        if self._started:
            raise NavigatorError("the pipeline was already started")
        self._require_active_run()
        self._started = True
        turn = self._new_turn()
        self._emit_turn_started(turn)
        out = self.require_and_reserve_next()
        self._drain_transitions()
        if out.kind is SupervisorOutcome.COMPLETE:
            # a definition with no required questions at all
            return self._finish_complete(turn, (), False)
        if out.kind is not SupervisorOutcome.CONTINUE:
            raise NavigatorError(
                f"unexpected supervisor outcome on start: {out.kind}"
            )
        question_id = out.next_question
        if question_id is None:
            raise NavigatorError("no opening question (internal error)")
        return self._deliver_reserved_question(
            turn, question_id, ContentKind.QUESTION
        )

    def receive_input(
        self,
        content: str,
        modality: Modality | None = None,
        audio_ref: str | None = None,
    ) -> TurnOutcome:
        """
        One ANSWER turn: the participant's input (the typed text, or
        the transcript of a voice input — the STT runs UPSTREAM, in
        app.py) runs through the full pipeline shape:

            Monitor (signals + intent) -> failure-choice check
            -> abort check -> the deterministic skip-REQUEST pre-checks
            (explicit skip / refusal: the question is probed first —
            ask why + alternatives — and skipped only once the
            skip-probe threshold is reached) -> Diagnostiker (the
            failure policy on a technical error) -> the deterministic
            steering decision -> Navigator -> Sprecher -> Validator
            -> Output

        and the outcome carries the released content (the next
        question, the follow-up re-ask, the closing or the
        out-of-scope hand-over — or the pre-approved failure/pause
        texts of the technical-failure policy) plus the fired
        intervention signals (the three SEPARATE typed signals — never
        merged). See the module docstring for the WInf-inspired
        steering / skip / failure behavior.
        """
        self._require_active_run()
        awaiting = self._run.awaiting_answer()
        if awaiting is None:
            raise NavigatorError(
                "no question is awaiting an answer — start() the "
                "pipeline first"
            )
        question_id = awaiting
        turn = self._new_turn()
        self._emit_turn_started(turn)
        session_modality = modality if modality is not None else self._modality
        self._emit_input(turn, content, session_modality, audio_ref)
        # 1. the Interventionsmonitor — checked BEFORE regular
        #    interview processing (plan §1.1); the three signal
        #    classes stay separate. (WInf-inspired: the monitor also
        #    measures the input's INTENT — the steering reactions
        #    below are driven by it, the Diagnostiker only evaluates
        #    the answer against the delivered question.)
        monitor_result = self._monitor.check(self.run_id, turn, content)
        signals = tuple(monitor_result.signals)
        intent = monitor_result.intent
        for signal in signals:
            self._emit_intervention(turn, signal)
        # the safety protocol: an EARLY_STOP suicidality signal ends
        # the run (the audit log is KEPT).
        for signal in signals:
            if (
                isinstance(signal, SuicidalitySignal)
                and signal.action is SuicidalityAction.EARLY_STOP
            ):
                return self._finish_safety_stop(turn, signals, False)
        # (WInf-inspired) the failure-choice protocol: the previous
        # turn ended at/over the technical-failure threshold, so THIS
        # input is the participant's continue/pause decision —
        # determined deterministically (no LLM).
        if self._failure_choice_pending:
            return self._handle_failure_choice(turn, content, signals)
        # 2. the abort check: an AbortRequestSignal does NOT stop the
        #    interview — it only informs the participant that they may
        #    abort at any time (the UI highlights the abort button;
        #    abort_interview() / abort_study() apply the
        #    participant's decision). A PromptInjectionSignal is an
        #    additional detection layer only (logged, no reaction).
        abort_available = any(
            isinstance(s, AbortRequestSignal) for s in signals
        )
        # (WInf-inspired) the DETERMINISTIC steering pre-checks — they
        # run BEFORE the Diagnostiker (they need no LLM at all): the
        # explicit skip command and the refusal intent are SKIP
        # REQUESTS — the question is NOT skipped immediately, the
        # system ASKS the participant why they want to skip (and
        # offers alternatives: a vague/short answer is fine) and skips
        # the CURRENT question (DELIVERED -> SKIPPED — and a skipped
        # question is never a negative answer) only once the
        # skip-probe threshold (max_skip_probes) is reached.
        if is_explicit_skip(content, self._steering.skip_phrases):
            return self._handle_skip_request(
                turn,
                question_id,
                SkipReason.EXPLICIT,
                trigger="the participant explicitly asked to skip the question",
                signals=signals,
                abort_available=abort_available,
            )
        if intent is InputIntent.REFUSAL:
            return self._handle_skip_request(
                turn,
                question_id,
                SkipReason.REFUSED,
                trigger="the participant refused to answer (intent: refusal)",
                signals=signals,
                abort_available=abort_available,
            )
        # 3. the Diagnostiker (the answer evaluation, as far as the
        #    steering decision requires). A TECHNICAL failure of the
        #    agent call (model error, invalid/missing structured
        #    output — the Phase 3 typed StructuredOutputError) is
        #    NEVER a participant action: the failure policy applies
        #    (WInf-inspired — no state change, no budget consumed,
        #    the participant is asked to resend; after the threshold
        #    the participant chooses continue/pause).
        candidates = self.candidate_set(turn)
        try:
            assessment = self._diagnostiker.assess(
                self.run_id, turn, question_id, content, candidates
            )
        except (NavigatorError, IllegalTransitionError):
            # internal pipeline errors fail loudly — they are not
            # technical (model) failures the participant can fix.
            raise
        except Exception as exc:  # noqa: BLE001 — the failure policy
            # deliberately broad: any agent-side failure (provider,
            # timeout, parse) is a technical failure.
            return self._handle_model_failure(
                turn, question_id, signals, abort_available, exc
            )
        self._emit_agent_output(turn, AgentName.DIAGNOSTIKER, assessment)
        # a successful agent call restarts the failure count.
        self._consecutive_model_failures = 0
        readings = [
            ValueReading(name=name, value=value, reliable=True)
            for name, value in assessment.navigation_values.items()
        ]

        # (WInf-inspired) the DETERMINISTIC steering decision: the
        # reaction is a function of (intent, clarity, the per-kind
        # re-ask counters, the SteeringConfig thresholds) — no LLM.
        steering = steering_decision(
            intent,
            assessment.answer_clarity,
            self._reask_counts.get(question_id, {}),
            self._steering,
        )
        if steering.action is SteeringAction.SKIP:
            # the budget was exhausted: the question is skipped as
            # unresolved — never a negative answer (see
            # _skip_current_question).
            return self._skip_current_question(
                turn,
                question_id,
                steering.skip_reason or SkipReason.UNRESOLVED,
                reason=steering.reason,
                signals=signals,
                abort_available=abort_available,
            )
        if steering.action is SteeringAction.REASK:
            # the answer lacks information for this kind: the
            # question is REVISED and re-asked (FOLLOWUP) — the
            # Navigator must not move on to a new question while one
            # is REVISED. The kind owns its counter (the per-kind
            # attempt counter; the budget is consumed here).
            kind = steering.reask_kind or ReaskKind.CLARIFICATION
            counts = self._reask_counts.setdefault(question_id, {})
            counts[kind.value] = counts.get(kind.value, 0) + 1
            self.apply_answer(
                question_id,
                readings,
                closed=False,
                cause=f"steering: re-ask ({kind.value}) — {steering.reason}",
            )
            self._run.revise_question(
                question_id,
                f"steering: re-ask kind={kind.value} "
                f"(attempt {counts[kind.value]}) — {steering.reason}",
            )
            out = self.require_and_reserve_next()
            self._drain_transitions()
            if (
                out.kind is not SupervisorOutcome.CONTINUE
                or out.next_question is None
            ):
                raise NavigatorError(
                    "unexpected supervisor outcome after a revised "
                    "question: " + str(out.kind)
                )
            result = self._deliver_reserved_question(
                turn, out.next_question, ContentKind.FOLLOWUP
            )
            return replace(
                result,
                signals=tuple(signals),
                abort_available=abort_available,
            )
        # 4. PROCEED — apply the answer (record + close + the online
        #    criteria + the group consequences — the single writer).
        #    (WInf-inspired) the question leaves the in-flight state:
        #    its per-kind re-ask counters are reset.
        self._reask_counts.pop(question_id, None)
        out = self.apply_answer(
            question_id,
            readings,
            closed=True,
            cause="diagnostiker: answer recorded",
        )
        self._drain_transitions()
        # the Diagnostiker's explicit criterion updates (the LLM path
        # — empty for the stub): applied after the deterministic
        # online proposals.
        for update in assessment.criterion_updates:
            update_cause = (
                f"diagnostiker: {update.reasoning}"
                if update.reasoning
                else (
                    "diagnostiker: "
                    + update.criterion_id
                    + " (evidence: "
                    + (update.evidence or "-")
                    + ")"
                )
            )
            self._run.set_criterion(
                update.criterion_id, update.status, update_cause
            )
        self._drain_transitions()
        # the next-question choice: MUST be one of the candidates of
        # the Navigator's live candidate set (the restriction the
        # pipeline enforces — contracts.NavigationAssessment); the
        # 2.5 skeleton always asks the DETERMINISTIC next question
        # (the agent's own choice among the candidates is wired with
        # the Phase 3 LLM).
        if assessment.next_question_id is not None:
            live = self.candidate_set(turn)
            if live.get(assessment.next_question_id) is None:
                raise NavigatorError(
                    "diagnostiker chose a question that is not in the "
                    "candidate set: " + assessment.next_question_id
                )
            if live.candidates[0].question_id != assessment.next_question_id:
                raise NavigatorError(
                    "the step 2.5 skeleton asks only the deterministic "
                    "next question; the agent choice is wired in "
                    "Phase 3"
                )
        return self._advance_after_answer(
            turn, out, signals, abort_available
        )

    def _handle_skip_request(
        self,
        turn_id: str,
        question_id: str,
        skip_reason: SkipReason,
        trigger: str,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        (WInf-inspired) the SKIP-REQUEST reaction — the explicit skip
        command (the deterministic command check) or the refusal
        intent (the monitor). The question is NOT skipped
        immediately: while the skip-probe budget (SteeringConfig.
        max_skip_probes, per question) still allows, the system ASKS
        the participant WHY they want to skip and offers alternatives
        (a vague/short answer is fine — the pre-approved probe text):
          * NO readings are recorded (the skip request is not an
            answer — a skip is never a negative answer);
          * the question is REVISED and re-asked (FOLLOWUP) with the
            probe text appended to the question text (DELIVERED ->
            REVISED -> DELIVERED — the participant can still answer
            the question after the probe, e.g. vaguely/shortly);
          * only once the probe threshold is reached the question is
            SKIPPED (DELIVERED -> SKIPPED, the skip_reason + the
            trigger audited) and the interview advances.
        """
        decision = skip_request_decision(
            self._reask_counts.get(question_id, {}),
            self._steering,
            skip_reason,
        )
        if decision.action is SteeringAction.SKIP:
            return self._skip_current_question(
                turn_id,
                question_id,
                skip_reason,
                reason=f"{trigger} — {decision.reason}",
                signals=signals,
                abort_available=abort_available,
            )
        kind = ReaskKind.SKIP_PROBE
        counts = self._reask_counts.setdefault(question_id, {})
        counts[kind.value] = counts.get(kind.value, 0) + 1
        # the question is re-offered WITH THE PROBE while no answer is
        # recorded yet (DELIVERED -> REVISED — the WInf-inspired
        # skip-probe re-ask; the audit cause names the kind + attempt):
        self._run.revise_question(
            question_id,
            f"steering: re-ask kind={kind.value} (attempt "
            f"{counts[kind.value]}) — {trigger}",
        )
        out = self.require_and_reserve_next()
        self._drain_transitions()
        if out.kind is not SupervisorOutcome.CONTINUE or out.next_question is None:
            raise NavigatorError(
                "unexpected supervisor outcome after a skip-probe re-ask: "
                + str(out.kind)
            )
        result = self._deliver_reserved_question(
            turn_id,
            out.next_question,
            ContentKind.FOLLOWUP,
            supplementary=self._steering.skip_probe_text,
        )
        return replace(
            result, signals=tuple(signals), abort_available=abort_available
        )

    def _skip_current_question(
        self,
        turn_id: str,
        question_id: str,
        skip_reason: SkipReason,
        reason: str,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        (WInf-inspired) the DETERMINISTIC skip reaction: the CURRENT
        (DELIVERED) question is locked as SKIPPED (DELIVERED ->
        SKIPPED) and the interview advances to the next question —
        the skip never ends the run.

        A SKIPPED QUESTION IS NEVER A NEGATIVE ANSWER: no value
        readings are recorded, no criteria are evaluated against the
        skip — the criteria that depend on the question stay
        OPEN/UNCLEAR exactly as they were (the participant's raw input
        of course remains in the audit log via the input events). The
        audit cause carries the WInf skip reason (explicit / refused /
        unresolved).
        """
        self._run.skip_question(
            question_id,
            f"steering: skipped ({skip_reason.value}) — {reason}",
        )
        # the question leaves the in-flight state:
        self._reask_counts.pop(question_id, None)
        self._consecutive_model_failures = 0
        self._drain_transitions()
        # a skip can COMPLETE the question's group (SKIPPED is a
        # terminal question state for the completion check) — then the
        # group's DEFERRED consequences (jumps, out-of-scope exit,
        # episode re-entry) must be resolved exactly as after a close.
        group = self._group_of[question_id]
        context = self._run.active_episode
        if (
            self._navigator.group_is_complete(self._run, group.id)
            and (group.id, context) not in self._completed_groups
        ):
            self._completed_groups.add((group.id, context))
            out = self._apply_group_completion(
                question_id,
                group,
                f"steering: skip ({skip_reason.value})",
            )
        else:
            out = self.require_and_reserve_next()
        return self._advance_after_answer(
            turn_id, out, signals, abort_available
        )

    def _handle_model_failure(
        self,
        turn_id: str,
        question_id: str,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
        exc: Exception,
    ) -> TurnOutcome:
        """
        (WInf-inspired) the TECHNICAL-FAILURE policy: the agent call
        (the Diagnostiker for now — Phase 3 extends the same policy to
        the Sprecher/Validator calls) failed technically. This is
        NEVER a participant action:
          * NO state change — the question stays DELIVERED (still
            awaiting its answer), nothing is recorded, the
            interview does not advance;
          * NO budget consumed — the per-kind re-ask counters are
            untouched;
          * the failure is logged (the model_failure audit event with
            the agent and the failure message) and the participant is
            asked to RESEND the answer (the pre-approved fixed text,
            delivered like the terminal texts);
          * after the configured number of CONSECUTIVE failures the
            PARTICIPANT CHOOSES: continue (the counter resets) or
            pause the run (RunStatus.PAUSED — frozen, resumable via
            resume_run()). The choice is determined on the NEXT input
            (the deterministic continue/pause command check).
        """
        self._consecutive_model_failures += 1
        self._emit(
            ModelFailureEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                agent=AgentName.DIAGNOSTIKER,
                detail=str(exc),
            )
        )
        threshold = self._steering.pause_after_model_failures
        if self._consecutive_model_failures >= threshold:
            # the threshold is reached: the participant now decides.
            self._failure_choice_pending = True
            text = self._steering.failure_choice_text
            reason = (
                f"technical failure {self._consecutive_model_failures} "
                f"of {threshold} — the participant chooses (continue / "
                "pause the run)"
            )
        else:
            text = self._steering.technical_failure_text
            reason = (
                f"technical failure {self._consecutive_model_failures} "
                f"of {threshold} — the participant is asked to resend "
                "the answer (no state change, no budget consumed)"
            )
        return self._deliver_terminal(
            turn_id,
            ContentKind.FEEDBACK,
            text,
            signals=signals,
            abort_available=abort_available,
            reason=reason,
            model_failure=True,
            consecutive_model_failures=self._consecutive_model_failures,
            failure_choice_pending=self._failure_choice_pending,
        )

    def _handle_failure_choice(
        self,
        turn_id: str,
        content: str,
        signals: tuple[InterventionSignal, ...],
    ) -> TurnOutcome:
        """
        (WInf-inspired) the CONTINUE / PAUSE decision after the
        technical-failure threshold: a deterministic, LLM-free command
        check (normalized exact match against the configured
        continue/pause phrases):
          * continue — the failure counter restarts and the pending
            answer is retried (the question stays DELIVERED; the
            participant simply resends it);
          * pause    — the run is PAUSED (ACTIVE -> PAUSED, the state
            transition is logged and the snapshot saved); the pending
            question stays DELIVERED — resume_run() re-asks it;
          * neither  — the choice is re-delivered (still pending, no
            state change).
        """
        normalized = normalize_command(content)
        continue_set = {
            normalize_command(phrase)
            for phrase in self._steering.continue_phrases
            if (phrase or "").strip()
        }
        pause_set = {
            normalize_command(phrase)
            for phrase in self._steering.pause_phrases
            if (phrase or "").strip()
        }
        if normalized and normalized in continue_set:
            self._failure_choice_pending = False
            self._consecutive_model_failures = 0
            return self._deliver_terminal(
                turn_id,
                ContentKind.FEEDBACK,
                self._steering.continue_confirmation_text,
                signals=signals,
                reason=(
                    "failure choice: continue — the pending answer is "
                    "retried (the failure counter restarts)"
                ),
            )
        if normalized and normalized in pause_set:
            self._failure_choice_pending = False
            self._run.pause_run(
                "failure choice: the participant pauses the interview"
            )
            self._drain_transitions()
            return self._deliver_terminal(
                turn_id,
                ContentKind.FEEDBACK,
                self._steering.pause_confirmation_text,
                signals=signals,
                reason="failure choice: the run is PAUSED (resumable)",
            )
        # neither command: the choice is still pending.
        return self._deliver_terminal(
            turn_id,
            ContentKind.FEEDBACK,
            self._steering.failure_choice_text,
            signals=signals,
            reason=(
                "failure choice still pending — the choice is "
                "re-delivered"
            ),
            failure_choice_pending=True,
        )

    def resume_run(self) -> TurnOutcome:
        """
        (WInf-inspired) resume a PAUSED run (the participant's
        explicit decision — e.g. in a later session): PAUSED ->
        ACTIVE (the state transition is logged), and the pending
        question (still DELIVERED — its answer was never recorded) is
        re-delivered through the Sprecher -> Validator -> Output chain
        as a pure output-layer retransmission (NO question-state
        transition — the question stays DELIVERED). The consecutive
        failure counter restarts. Only possible on a PAUSED run.
        """
        if self._run.status is not RunStatus.PAUSED:
            raise NavigatorError(
                f"resume_run() is only possible on a PAUSED run (the "
                f"run is {self._run.status.value})"
            )
        self._run.resume_run(
            "supervisor: the participant resumes the paused interview"
        )
        self._drain_transitions()
        self._consecutive_model_failures = 0
        self._failure_choice_pending = False
        question_id = self._run.awaiting_answer()
        if question_id is None:
            raise NavigatorError(
                "no question is awaiting an answer on the paused run "
                "(internal error — a pause only happens while a "
                "question is DELIVERED)"
            )
        turn = self._new_turn()
        self._emit_turn_started(turn)
        return self._deliver_reserved_question(
            turn, question_id, ContentKind.QUESTION, advance_state=False
        )

    def _advance_after_answer(
        self,
        turn_id: str,
        out: SupervisorOutcome,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        Finish one answer turn after the answer (or the skip) was
        applied: handle the run-level outcomes (out-of-scope exit,
        completion, the pending episode decision), otherwise reserve
        and deliver the deterministic next question.
        """
        if out.kind is SupervisorOutcome.OUT_OF_SCOPE:
            return self._finish_out_of_scope_turn(
                turn_id, signals, abort_available
            )
        if out.kind is SupervisorOutcome.COMPLETE:
            return self._finish_complete(turn_id, signals, abort_available)
        if out.kind is SupervisorOutcome.PENDING_EPISODE:
            self._pending_source = out.pending_source
            return TurnOutcome(
                turn_id=turn_id,
                run_status=self.status,
                signals=tuple(signals),
                abort_available=abort_available,
                pending_episode=True,
                reason=out.message
                or (
                    "a dynamic episode jump awaits the Diagnostiker's "
                    "EpisodeDecision (complete_pending_episode)"
                ),
            )
        out = self.require_and_reserve_next()
        self._drain_transitions()
        if out.kind is SupervisorOutcome.COMPLETE:
            return self._finish_complete(turn_id, signals, abort_available)
        if out.kind is not SupervisorOutcome.CONTINUE or out.next_question is None:
            raise NavigatorError(
                "unexpected supervisor outcome after the answer: "
                + str(out.kind)
            )
        result = self._deliver_reserved_question(
            turn_id, out.next_question, ContentKind.QUESTION
        )
        return replace(
            result,
            signals=tuple(signals),
            abort_available=abort_available,
        )

    def abort_interview(self, feedback: str | None = None) -> TurnOutcome:
        """
        The participant's decision to abort the INTERVIEW only (via
        the UI): the run ends with
        RunAbortCause.PARTICIPANT_ABORT_INTERVIEW, the pre-approved
        abort closing is delivered, and the audit log is KEPT
        (optionally with the participant's free-text feedback from the
        frontend text field).
        """
        self._require_active_run()
        self._run.abort_run(
            "participant: abort the interview (via the UI)"
        )
        self._drain_transitions()
        self._emit_run_aborted(
            self._current_turn_id,
            RunAbortCause.PARTICIPANT_ABORT_INTERVIEW,
            "participant: abort the interview",
            feedback=feedback,
        )
        return self._deliver_terminal(
            None,
            ContentKind.ABORT_CLOSING,
            self._abort_closing_text,
            reason="participant aborted the interview",
        )

    def abort_study(self) -> TurnOutcome:
        """
        The participant's decision to abort the WHOLE STUDY (via the
        UI): the run ends with
        RunAbortCause.PARTICIPANT_ABORT_STUDY and — when a repository
        was configured — the complete run data (the participant's
        data) is deleted via the store's abort_study(): only the
        tombstone (run_id, aborted_at, cause) remains. Nothing further
        is delivered (the session ends).
        """
        self._require_active_run()
        self._run.abort_run(
            "participant: abort the whole study (via the UI)"
        )
        self._drain_transitions()
        self._emit_run_aborted(
            self._current_turn_id,
            RunAbortCause.PARTICIPANT_ABORT_STUDY,
            "participant: abort the whole study",
        )
        if self._repo is not None:
            self._repo.abort_study(self.run_id)
            self._study_aborted = True
        return TurnOutcome(
            turn_id=None,
            run_status=self.status,
            reason="participant aborted the study",
        )

    def complete_pending_episode(
        self, decision: EpisodeDecision
    ) -> TurnOutcome:
        """
        The LLM-driven episode mode (auto_episode_decisions=False):
        apply the Diagnostiker's EpisodeDecision to the PENDING
        dynamic episode jump (pending_source) and continue — the
        re-asked question (or the completion) goes through the full
        Sprecher -> Validator -> Output chain.
        """
        self._require_active_run()
        if self._pending_source is None:
            raise NavigatorError("no episode decision is pending")
        source = self._pending_source
        self._pending_source = None
        turn = self._new_turn()
        self._emit_turn_started(turn)
        out = self.apply_episode_decision(source, decision)
        self._drain_transitions()
        if out.kind is SupervisorOutcome.OUT_OF_SCOPE:
            return self._finish_out_of_scope_turn(turn, (), False)
        if out.kind is SupervisorOutcome.COMPLETE:
            return self._finish_complete(turn, (), False)
        if out.kind is SupervisorOutcome.PENDING_EPISODE:
            # a chained jump left a NEW pending episode
            self._pending_source = out.pending_source
            return TurnOutcome(
                turn_id=turn,
                run_status=self.status,
                pending_episode=True,
                reason=out.message
                or "another dynamic episode jump is pending",
            )
        out = self.require_and_reserve_next()
        self._drain_transitions()
        if out.kind is SupervisorOutcome.COMPLETE:
            return self._finish_complete(turn, (), False)
        if out.kind is not SupervisorOutcome.CONTINUE or out.next_question is None:
            raise NavigatorError(
                "unexpected supervisor outcome after the episode "
                "decision: " + str(out.kind)
            )
        return self._deliver_reserved_question(
            turn, out.next_question, ContentKind.QUESTION
        )

    # ------------------------------------------------------------------
    # the pipeline's audit events (the single writer logs EVERYTHING)
    # ------------------------------------------------------------------

    def _emit(self, event: RunEvent) -> None:
        """Append one event to the in-memory stream (and the store)."""
        self._events.append(event)
        if self._events_store is not None:
            self._events_store.append(event)

    def _drain_transitions(self) -> None:
        """
        Emit the STATE_TRANSITION events of every run transition made
        since the last drain (in order) — the transition log the
        state machine records (TransitionRecord.to_event()). The
        pipeline calls this at its step boundaries, so every state
        change of every supervisor call is logged exactly once.
        """
        records = self._run.transitions
        for record in records[self._emitted_transitions :]:
            event = record.to_event(
                self.run_id, self._definition.definition_version
            )
            # the state machine's to_event() leaves turn_id unset
            # (transitions span turn boundaries); the Supervisor —
            # the single writer — stamps the CURRENT turn onto each
            # event so the audit stream is filterable per turn:
            if self._current_turn_id is not None:
                event = event.model_copy(
                    update={"turn_id": self._current_turn_id}
                )
            self._emit(event)
        self._emitted_transitions = len(records)

    def _emit_run_started(self) -> None:
        self._emit(
            RunStartedEvent(
                run_id=self.run_id,
                definition_version=self._definition.definition_version,
                study_condition=self._study_condition,
            )
        )

    def _emit_turn_started(self, turn_id: str) -> None:
        self._emit(
            TurnStartedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
            )
        )

    def _emit_turn_completed(self, turn_id: str) -> None:
        self._emit(
            TurnCompletedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
            )
        )

    def _emit_input(
        self,
        turn_id: str,
        content: str,
        modality: Modality,
        audio_ref: str | None,
    ) -> None:
        self._emit(
            InputReceivedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                modality=modality,
                content=content if modality is Modality.TEXT else None,
                audio_ref=audio_ref if modality is Modality.VOICE else None,
            )
        )
        if modality is Modality.VOICE:
            # the transcript is the (normalized) input the Monitor
            # checks and the Diagnostiker evaluates (voice only).
            self._emit(
                InputTranscribedEvent(
                    run_id=self.run_id,
                    turn_id=turn_id,
                    definition_version=self._definition.definition_version,
                    transcript=content,
                    language=self._target_language(),
                )
            )

    def _emit_intervention(
        self, turn_id: str, signal: InterventionSignal
    ) -> None:
        self._emit(
            InterventionSignalEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                signal=signal,
            )
        )

    def _emit_agent_output(
        self, turn_id: str, agent: AgentName, payload: Any
    ) -> None:
        """
        One LLM agent output, raw + parsed. The stubs produce no raw
        text (raw_output=None); the parsed structured output is the
        boundary model itself (JSON-serializable by construction).
        """
        self._emit(
            AgentOutputEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                agent=agent,
                raw_output=None,
                parsed=payload.model_dump(mode="json"),
            )
        )

    def _emit_validation(
        self, turn_id: str, decision: ValidationDecision
    ) -> None:
        self._emit(
            ValidationDecisionEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                decision=decision,
                raw_output=None,
            )
        )

    def _emit_run_completed(self, turn_id: str | None, reason: str) -> None:
        self._emit(
            RunCompletedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                reason=reason,
            )
        )

    def _emit_run_aborted(
        self,
        turn_id: str | None,
        cause: RunAbortCause,
        reason: str,
        feedback: str | None = None,
    ) -> None:
        self._emit(
            RunAbortedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                cause=cause,
                reason=reason,
                feedback=feedback,
            )
        )

    # ------------------------------------------------------------------
    # the pipeline's delivery (Sprecher -> Validator -> Output)
    # ------------------------------------------------------------------

    def _target_language(self) -> str:
        if self._study_condition is not None:
            return self._study_condition.interview_language
        return "de"

    def _new_turn(self) -> str:
        self._turn_sequence += 1
        self._current_turn_id = make_turn_id(self._turn_sequence)
        return self._current_turn_id

    def _next_delivery_id(self) -> str:
        self._delivery_sequence += 1
        return delivery_id(self.run_id, self._delivery_sequence)

    def _save_state(self) -> None:
        """
        Persist the state snapshot (run_version +1 — the store
        rejects anything else: the single-writer discipline). The
        Supervisor owns the version even without a repository (it
        is the seed of the optimistic run_version check, Phase
        5.5); only the write itself is optional.
        """
        if self._study_aborted:
            return
        self._run_version += 1
        if self._repo is not None:
            self._repo.save_state(
                self.run_id, self._run.snapshot(self._run_version)
            )

    def _require_active_run(self) -> None:
        if self._run.status is not RunStatus.ACTIVE:
            if self._run.status is RunStatus.PAUSED:
                raise NavigatorError(
                    "the run is paused — resume_run() first (the "
                    "participant's explicit decision)"
                )
            raise NavigatorError(
                f"the run is {self._run.status.value} — no further "
                f"pipeline steps are possible"
            )

    def _deliver_reserved_question(
        self,
        turn_id: str,
        question_id: str,
        kind: ContentKind,
        advance_state: bool = True,
        supplementary: str | None = None,
    ) -> TurnOutcome:
        """
        Release the reserved question through Sprecher -> Validator
        -> Output (the deterministic question text with the {text
        tokens} rendered — OFQ-4/OFQ-5) and hand it to the output
        layer (the delivery lifecycle events are logged).

        `supplementary` (WInf-inspired skip probe): pre-approved
        text appended to the question text of a re-asked (REVISED)
        question — the probe asks the participant WHY they want to
        skip and offers alternatives (a vague/short answer is fine).
        With the Phase-3 LLM speaker the probe becomes part of the
        re-ask context instead (the same intent, generated wording).

        `advance_state=False` (WInf-inspired resume retransmission):
        the question is ALREADY DELIVERED (its answer is pending, e.g.
        after a pause) — it is re-delivered as a pure output-layer
        event with NO question-state transition (the question stays
        DELIVERED; DELIVERED -> DELIVERED is not a legal move).
        """
        text = self._navigator.rendered_text(question_id, self._run)
        if supplementary:
            text = f"{text}\n{supplementary}"
        package = ApprovedContentPackage(
            run_id=self.run_id,
            turn_id=turn_id,
            content=[
                ContentPiece(
                    kind=kind,
                    text=text,
                    source=ContentSource.GUIDE_VERBATIM,
                )
            ],
            invariants=list(InvariantAspect),
            target_language=self._target_language(),
        )
        speaker_output = self._sprecher.speak(package)
        self._emit_agent_output(turn_id, AgentName.SPRECHER, speaker_output)
        released, decision = self._validate_loop(
            turn_id, package, speaker_output
        )
        release_id = self._next_delivery_id()
        self._emit(
            DeliveryReservedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=release_id,
                content_kind=released.kind,
                content_text=released.text,
            )
        )
        self._emit(
            DeliveryValidatedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=release_id,
                outcome=decision.outcome,
            )
        )
        # the single writer: the reserved (or REVISED re-asked)
        # question becomes DELIVERED — skipped for the resume
        # retransmission (the question is already DELIVERED).
        if advance_state:
            self.deliver(
                f"output layer: {question_id} sent"
                if kind is ContentKind.QUESTION
                else f"output layer: {question_id} re-asked (follow-up)"
            )
            self._drain_transitions()
        delivery = Delivery(
            run_id=self.run_id,
            delivery_id=release_id,
            turn_id=turn_id,
            modality=self._modality,
            content=released,
            question_id=question_id,
        )
        self._output_port.deliver(delivery)
        self._emit(
            DeliveryDeliveredEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=release_id,
                modality=self._modality,
            )
        )
        self._emit_turn_completed(turn_id)
        self._save_state()
        return TurnOutcome(
            turn_id=turn_id,
            run_status=self.status,
            delivered=released,
            delivery_id=release_id,
            question_id=question_id,
        )

    def _validate_loop(
        self,
        turn_id: str,
        package: ApprovedContentPackage,
        speaker_output: SpeakerOutput,
    ) -> tuple[ContentPiece, ValidationDecision]:
        """
        The Validator's bounded revision loop (the shape, step 2.5):
        up to `max_validation_rounds` rounds of Speaker -> Validator.
        ALLOW releases the version; BLOCK — or the rounds exhausted
        after REVISE — fails CLOSED to the pre-approved CANONICAL
        FALLBACK (the documented canonical bypass: a session can
        always complete).
        """
        current = speaker_output
        for round_no in range(1, self._max_validation_rounds + 1):
            decision = self._validator.validate(current, self._study_condition)
            self._emit_validation(turn_id, decision)
            if decision.outcome is ValidationOutcome.ALLOW:
                return current.content[0], decision
            if decision.outcome is ValidationOutcome.BLOCK:
                return self._canonical_fallback(), decision
            # REVISE: the instructions carry into the next Speaker
            # round (bounded — see above).
            if round_no < self._max_validation_rounds:
                current = self._sprecher.speak(
                    package, decision.revision_instructions
                )
                self._emit_agent_output(turn_id, AgentName.SPRECHER, current)
        # the rounds exhausted (every round REVISEd): fail-closed.
        exhausted = ValidationDecision(
            run_id=self.run_id,
            turn_id=turn_id,
            outcome=ValidationOutcome.BLOCK,
            reasons=[
                ValidationReason(
                    code="VALIDATION_ROUNDS_EXHAUSTED",
                    detail=(
                        f"{self._max_validation_rounds} revision rounds "
                        "exhausted — the canonical fallback is delivered "
                        "instead"
                    ),
                )
            ],
            checked_aspects=list(ValidationAspect),
            revision_instructions=None,
            canonical_fallback_used=True,
        )
        self._emit_validation(turn_id, exhausted)
        return self._canonical_fallback(), exhausted

    def _canonical_fallback(self) -> ContentPiece:
        """The pre-approved canonical fallback (the documented bypass)."""
        message = (self._definition.canonical_fallback_message or "").strip()
        if not message:
            raise NavigatorError(
                "the validator BLOCKED the content but the definition "
                "has no canonical fallback message — a session must "
                "always be able to complete"
            )
        return ContentPiece(
            kind=ContentKind.CANONICAL_FALLBACK,
            text=message,
            source=ContentSource.PRE_APPROVED_FALLBACK,
        )

    def _deliver_terminal(
        self,
        turn_id: str | None,
        kind: ContentKind,
        text: str,
        *,
        signals: tuple[InterventionSignal, ...] = (),
        abort_available: bool = False,
        out_of_scope: OutOfScopeExit | None = None,
        reason: str = "",
        # (WInf-inspired) the technical-failure policy fields:
        model_failure: bool = False,
        consecutive_model_failures: int = 0,
        failure_choice_pending: bool = False,
    ) -> TurnOutcome:
        """
        Hand a PRE-APPROVED (PRE_APPROVED_FALLBACK) terminal piece
        (closing / abort closing / out-of-scope hand-over / safety
        hint) over to the output layer with the delivery lifecycle
        events. Terminal content bypasses the Speaker and the
        Validator by design — the binding exit control applies to the
        Speaker's versions (the canonical fallback is the same class).
        TURN_COMPLETED is emitted only when a turn was started for it
        (the participant aborts deliver outside any answer turn).
        """
        delivery_id = self._next_delivery_id()
        piece = ContentPiece(
            kind=kind, text=text, source=ContentSource.PRE_APPROVED_FALLBACK
        )
        self._emit(
            DeliveryReservedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=delivery_id,
                content_kind=kind,
                content_text=text,
            )
        )
        self._emit(
            DeliveryValidatedEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=delivery_id,
                outcome=ValidationOutcome.ALLOW,
            )
        )
        delivery = Delivery(
            run_id=self.run_id,
            delivery_id=delivery_id,
            turn_id=turn_id,
            modality=self._modality,
            content=piece,
            question_id=None,
        )
        self._output_port.deliver(delivery)
        self._emit(
            DeliveryDeliveredEvent(
                run_id=self.run_id,
                turn_id=turn_id,
                definition_version=self._definition.definition_version,
                delivery_id=delivery_id,
                modality=self._modality,
            )
        )
        if turn_id is not None:
            self._emit_turn_completed(turn_id)
        self._save_state()
        return TurnOutcome(
            turn_id=turn_id,
            run_status=self.status,
            delivered=piece,
            delivery_id=delivery_id,
            signals=signals,
            abort_available=abort_available,
            out_of_scope=out_of_scope,
            model_failure=model_failure,
            consecutive_model_failures=consecutive_model_failures,
            failure_choice_pending=failure_choice_pending,
            reason=reason,
        )

    def _finish_complete(
        self,
        turn_id: str | None,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        The run was COMPLETED (inside require_and_reserve_next): log
        the run_completed event and deliver the pre-approved closing.
        """
        self._emit_run_completed(turn_id, CAUSE_RUN_COMPLETE_NO_QUESTIONS)
        return self._deliver_terminal(
            turn_id,
            ContentKind.CLOSING,
            self._closing_text,
            signals=signals,
            abort_available=abort_available,
            reason="the interview is complete",
        )

    def _finish_out_of_scope_turn(
        self,
        turn_id: str | None,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        The OUT-OF-SCOPE exit (OFQ-6): the run was ABORTED via
        abort_out_of_scope (the state transition is already logged) —
        log the run_aborted event and deliver the pre-approved
        hand-over (the external target the interview routes to, with
        the definition's own note, if any).
        """
        exit = self._run.out_of_scope
        assert exit is not None  # set by abort_out_of_scope
        self._emit_run_aborted(
            turn_id,
            RunAbortCause.OUT_OF_SCOPE,
            "out-of-scope exit: the interview continues with "
            + exit.target_ref
            + " (outside this definition)",
        )
        text = self._out_of_scope_template.format(target=exit.target_ref)
        if exit.note:
            text += " Hinweis: " + exit.note
        return self._deliver_terminal(
            turn_id,
            ContentKind.ABORT_CLOSING,
            text,
            signals=signals,
            abort_available=abort_available,
            out_of_scope=exit,
            reason="out-of-scope exit: the interview continues with "
            + exit.target_ref,
        )

    def _finish_safety_stop(
        self,
        turn_id: str,
        signals: tuple[InterventionSignal, ...],
        abort_available: bool,
    ) -> TurnOutcome:
        """
        The safety protocol (SuicidalityAction.EARLY_STOP): the
        diagnosis is stopped early — the run ends via
        RunAbortCause.SAFETY_EARLY_STOP, the audit log is KEPT
        (unlike a study-wide participant abort), and the participant
        is handed the calm pre-approved hint.
        """
        self._run.abort_run(
            "safety protocol: early stop (suicidality signal)"
        )
        self._drain_transitions()
        self._emit_run_aborted(
            turn_id,
            RunAbortCause.SAFETY_EARLY_STOP,
            "safety protocol: early stop (suicidality signal)",
        )
        return self._deliver_terminal(
            turn_id,
            ContentKind.FEEDBACK,
            self._safety_hint_text,
            signals=signals,
            abort_available=abort_available,
            reason="safety protocol: early stop",
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_active(self) -> bool:
        """Raise (returning True) when the run is no longer ACTIVE."""
        if self._run.status is not RunStatus.ACTIVE:
            raise NavigatorError(
                f"the run is {self._run.status.value} — no further steps "
                f"are possible"
            )
        return False

    def _ensure_required(self, question_id: str, cause: str) -> None:
        """Require an UNSEEN question (recording the ACTIVE episode)."""
        if self._run.question_state(question_id) is QuestionState.UNSEEN:
            self._run.require_question(
                question_id, cause, episode_id=self._run.active_episode
            )

    def _skip(self, question_id: str, cause: str) -> None:
        """Apply a skip proposal (UNSEEN -> REQUIRED -> SKIPPED)."""
        state = self._run.question_state(question_id)
        if state is QuestionState.UNSEEN:
            self._ensure_required(question_id, cause)
        if self._run.question_state(question_id) is QuestionState.REQUIRED:
            self._run.skip_question(question_id, cause)

    def _questions_of(self, target_ref: str) -> list[str]:
        """The questions of a target (question / group / section)."""
        group = self._definition.group_by_id(target_ref)
        if group is not None:
            return [q.id for q in group.questions]
        question = self._definition.question_by_id(target_ref)
        if question is not None:
            return [question.id]
        section = self._definition.section_by_id(target_ref)
        if section is not None:
            return self._navigator._section_questions(section)
        raise NavigatorError(
            f"unknown target {target_ref!r} (no question, group or "
            f"section)"
        )

    @staticmethod
    def _normalize_readings(
        readings: Mapping[str, ValueReading] | Sequence[ValueReading] | None,
    ) -> dict[str, ValueReading]:
        if not readings:
            return {}
        if isinstance(readings, Mapping):
            return dict(readings)
        return {reading.name: reading for reading in readings}
