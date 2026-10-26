"""
INTERVIEW RUN — THE DETERMINISTIC SESSION STATE MACHINE
=======================================================
The InterviewRun state machine — question states (UNSEEN / REQUIRED / RESERVED
/ DELIVERED / ANSWER_RECORDED / CLOSED / REVISED / SKIPPED) and criterion states
(OPEN / UNCLEAR / SATISFIED / NOT_SATISFIED, the four-valued logic), plus
the run-level status (ACTIVE / COMPLETED / ABORTED) — with explicit
transition rules and no other mutation path.

The state machine is PURE and deterministic: no LLM, no I/O, no Flask.
The only way a state changes is one of the transition methods below;
anything else raises IllegalTransitionError. Every applied transition is
appended to the transition log (the audit material the Supervisor writes
to the event store in step 2.5, via TransitionRecord.to_event()). The
persisted RunState (store.py) is the Supervisor's copy — the Supervisor
is the sole writer of that snapshot (run the snapshot() helper).

Transition tables (from -> legal to):

  RUN (RunStatus):
      ACTIVE -> COMPLETED | ABORTED
      COMPLETED, ABORTED are terminal
      (the ONLY way a run ends — no question-level transition,
       including a skip, ends the interview)
  QUESTION (QuestionState):
      UNSEEN           -> REQUIRED        (the Navigator marks it required)
      REQUIRED         -> RESERVED        (reserved as the next question)
                          SKIPPED         (never delivered: the participant
                                           already gave us this information;
                                           the criteria depending on it stay
                                           evaluable; the interview continues —
                                           see contracts.QuestionState)
      RESERVED         -> DELIVERED       (sent via the output layer)
      DELIVERED        -> ANSWER_RECORDED (the participant's answer is recorded)
      ANSWER_RECORDED  -> CLOSED          (interpreted; no information missing)
                          REVISED         (interpreted; information missing -> re-ask)
      REVISED          -> DELIVERED       (re-entry for the re-ask; the
                                           Navigator must NOT move on to a
                                           new question while one is REVISED)
      CLOSED, SKIPPED are terminal
  CRITERION (CriterionStatus):
      OPEN          -> SATISFIED | NOT_SATISFIED | UNCLEAR
      UNCLEAR       -> SATISFIED | NOT_SATISFIED | UNCLEAR
                      (a clarification may re-mark UNCLEAR)
      SATISFIED     -> UNCLEAR           (decided criteria can be reopened)
      NOT_SATISFIED -> UNCLEAR           (but NEVER directly to another
                                          decided state)

Invariants enforced beyond the tables:
  * at most one question in RESERVED at a time (the single next question)
  * once the run is COMPLETED/ABORTED, no further state changes at all
  * every transition carries a non-empty cause (the audit "why")
  * a skipped question does NOT end the interview: REQUIRED -> SKIPPED
    only locks that one question (the information was already known, so
    the criteria depending on it can be evaluated without delivery);
    the run stays ACTIVE

Used by: the Supervisor (Phase 2, step 2.5) and the tests (Phase 2,
step 2.6). The legacy pipeline does not use this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

from contracts import (
    RUN_ID_RE,
    CriterionStatus,
    QuestionState,
    RunState,
    RunStatus,
    StateTransitionEvent,
    SubjectKind,
    utc_now,
)
from interview_definition import InterviewDefinition


class IllegalTransitionError(RuntimeError):
    """
    A state transition outside the legal transition tables was attempted
    (or an invariant was violated, e.g. a second RESERVED question).
    """

    def __init__(
        self,
        subject: SubjectKind,
        subject_id: str,
        from_state: str,
        to_state: str,
        detail: str = "",
    ) -> None:
        if subject is SubjectKind.RUN:
            who = "the run"
        else:
            who = f"{subject.value} '{subject_id}'"
        message = f"illegal {subject.value} transition for {who}: {from_state} -> {to_state}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)
        self.subject = subject
        self.subject_id = subject_id
        self.from_state = from_state
        self.to_state = to_state


# ---------------------------------------------------------------------------
# the explicit transition tables (the ONLY legal moves)
# ---------------------------------------------------------------------------

_QUESTION_TRANSITIONS: Final[dict[QuestionState, frozenset[QuestionState]]] = {
    QuestionState.UNSEEN: frozenset({QuestionState.REQUIRED}),
    QuestionState.REQUIRED: frozenset({QuestionState.RESERVED, QuestionState.SKIPPED}),
    QuestionState.RESERVED: frozenset({QuestionState.DELIVERED}),
    QuestionState.DELIVERED: frozenset({QuestionState.ANSWER_RECORDED}),
    QuestionState.ANSWER_RECORDED: frozenset({QuestionState.CLOSED, QuestionState.REVISED}),
    QuestionState.REVISED: frozenset({QuestionState.DELIVERED}),
    QuestionState.CLOSED: frozenset(),
    QuestionState.SKIPPED: frozenset(),
}

_CRITERION_TRANSITIONS: Final[dict[CriterionStatus, frozenset[CriterionStatus]]] = {
    CriterionStatus.OPEN: frozenset(
        {CriterionStatus.SATISFIED, CriterionStatus.NOT_SATISFIED, CriterionStatus.UNCLEAR}
    ),
    CriterionStatus.UNCLEAR: frozenset(
        {CriterionStatus.SATISFIED, CriterionStatus.NOT_SATISFIED, CriterionStatus.UNCLEAR}
    ),
    CriterionStatus.SATISFIED: frozenset({CriterionStatus.UNCLEAR}),
    CriterionStatus.NOT_SATISFIED: frozenset({CriterionStatus.UNCLEAR}),
}

_RUN_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.ACTIVE: frozenset({RunStatus.COMPLETED, RunStatus.ABORTED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.ABORTED: frozenset(),
}


@dataclass(frozen=True)
class TransitionRecord:
    """
    One applied state transition — the in-memory audit entry. The
    Supervisor (step 2.5) persists these as StateTransitionEvent entries
    in the append-only event log (store.py) via to_event().
    """

    subject: SubjectKind
    subject_id: str  # "" for run-level transitions
    from_state: str  # plain strings: each subject has its own state vocabulary
    to_state: str
    cause: str
    timestamp: datetime

    def to_event(self, run_id: str, definition_version: str) -> StateTransitionEvent:
        """
        Build the audit event for this transition. event_id stays None
        until the EventStore assigns the gapless sequence number.
        """
        return StateTransitionEvent(
            run_id=run_id,
            definition_version=definition_version,
            subject=self.subject,
            subject_id=self.subject_id or None,
            from_state=self.from_state,
            to_state=self.to_state,
            cause=self.cause,
        )


def _require_cause(cause: str) -> str:
    """Every transition must carry a non-empty reason (the audit "why")."""
    if not isinstance(cause, str) or not cause.strip():
        raise ValueError(
            "cause must be a non-empty string (the audit log must be able "
            "to say WHY this state changed)"
        )
    return cause


class InterviewRun:
    """
    The deterministic state machine of one interview run.

    InterviewRun(definition, run_id) starts with the run ACTIVE, every
    question UNSEEN and every criterion OPEN. States change ONLY through
    the transition methods; question_states / criterion_states /
    transitions return copies, so external code can observe the state but
    not mutate it ("no other mutation path").
    """

    def __init__(self, definition: InterviewDefinition, run_id: str) -> None:
        if not RUN_ID_RE.match(run_id):
            raise ValueError(
                f"invalid run_id {run_id!r} (must match the run id scheme "
                f"from contracts.py, e.g. contracts.new_run_id())"
            )
        self._definition = definition
        self._run_id = run_id
        self._status: RunStatus = RunStatus.ACTIVE
        self._question_states: dict[str, QuestionState] = {
            q.id: QuestionState.UNSEEN for q in definition.all_questions()
        }
        self._criterion_states: dict[str, CriterionStatus] = {
            c.id: CriterionStatus.OPEN for c in definition.all_criteria()
        }
        self._transitions: list[TransitionRecord] = []

    # ------------------------------------------------------------------
    # internal helpers (the single choke point for every state change)
    # ------------------------------------------------------------------

    def _log(
        self,
        subject: SubjectKind,
        subject_id: str,
        from_state: str,
        to_state: str,
        cause: str,
    ) -> TransitionRecord:
        record = TransitionRecord(
            subject=subject,
            subject_id=subject_id,
            from_state=from_state,
            to_state=to_state,
            cause=cause,
            timestamp=utc_now(),
        )
        self._transitions.append(record)
        return record

    def _apply_question(
        self, question_id: str, target: QuestionState, cause: str
    ) -> TransitionRecord:
        self._require_active()
        question_id = self._known_question(question_id)
        current = self._question_states[question_id]
        legal = _QUESTION_TRANSITIONS[current]
        if target not in legal:
            legal_names = ", ".join(s.value for s in legal) or "none (terminal state)"
            raise IllegalTransitionError(
                SubjectKind.QUESTION,
                question_id,
                current.value,
                target.value,
                f"legal from {current.value}: {legal_names}",
            )
        self._question_states[question_id] = target
        return self._log(SubjectKind.QUESTION, question_id, current.value, target.value, cause)

    def _apply_criterion(
        self, criterion_id: str, target: CriterionStatus, cause: str
    ) -> TransitionRecord:
        self._require_active()
        criterion_id = self._known_criterion(criterion_id)
        current = self._criterion_states[criterion_id]
        legal = _CRITERION_TRANSITIONS[current]
        if target not in legal:
            legal_names = ", ".join(s.value for s in legal) or "none (terminal state)"
            raise IllegalTransitionError(
                SubjectKind.CRITERION,
                criterion_id,
                current.value,
                target.value,
                f"legal from {current.value}: {legal_names}",
            )
        self._criterion_states[criterion_id] = target
        return self._log(
            SubjectKind.CRITERION, criterion_id, current.value, target.value, cause
        )

    def _apply_run(self, target: RunStatus, cause: str) -> TransitionRecord:
        current = self._status
        legal = _RUN_TRANSITIONS[current]
        if target not in legal:
            legal_names = ", ".join(s.value for s in legal) or "none (terminal state)"
            raise IllegalTransitionError(
                SubjectKind.RUN,
                "",
                current.value,
                target.value,
                f"legal from {current.value}: {legal_names}",
            )
        self._status = target
        return self._log(SubjectKind.RUN, "", current.value, target.value, cause)

    def _require_active(self) -> None:
        """Question/criterion states may only change while the run is ACTIVE."""
        if self._status is not RunStatus.ACTIVE:
            raise IllegalTransitionError(
                SubjectKind.RUN,
                "",
                self._status.value,
                "TERMINAL",
                f"the run is {self._status.value} — no further state changes are allowed",
            )

    def _known_question(self, question_id: str) -> str:
        if question_id not in self._question_states:
            raise ValueError(
                f"unknown question id {question_id!r} (known: "
                f"{', '.join(self._question_states) or 'none'})"
            )
        return question_id

    def _known_criterion(self, criterion_id: str) -> str:
        if criterion_id not in self._criterion_states:
            raise ValueError(
                f"unknown criterion id {criterion_id!r} (known: "
                f"{', '.join(self._criterion_states) or 'none'})"
            )
        return criterion_id

    # ------------------------------------------------------------------
    # the ONLY mutation paths — question transitions
    # ------------------------------------------------------------------

    def require_question(self, question_id: str, cause: str) -> TransitionRecord:
        """UNSEEN -> REQUIRED: the Navigator marks the question as currently required."""
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.REQUIRED, cause)

    def reserve_question(self, question_id: str, cause: str) -> TransitionRecord:
        """REQUIRED -> RESERVED: reserved as the next question (at most one at a time)."""
        _require_cause(cause)
        reserved = [q for q, s in self._question_states.items() if s is QuestionState.RESERVED]
        if reserved:
            raise IllegalTransitionError(
                SubjectKind.QUESTION,
                question_id,
                QuestionState.REQUIRED.value,
                QuestionState.RESERVED.value,
                f"another question is already RESERVED ('{reserved[0]}') — "
                "at most one question may be reserved at a time",
            )
        return self._apply_question(question_id, QuestionState.RESERVED, cause)

    def deliver_question(self, question_id: str, cause: str) -> TransitionRecord:
        """RESERVED -> DELIVERED: the question was sent via the output layer."""
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.DELIVERED, cause)

    def record_answer(self, question_id: str, cause: str) -> TransitionRecord:
        """DELIVERED -> ANSWER_RECORDED: the participant's answer is recorded."""
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.ANSWER_RECORDED, cause)

    def close_question(self, question_id: str, cause: str) -> TransitionRecord:
        """ANSWER_RECORDED -> CLOSED: the answer is interpreted, no information missing."""
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.CLOSED, cause)

    def revise_question(self, question_id: str, cause: str) -> TransitionRecord:
        """ANSWER_RECORDED -> REVISED: the answer lacks information, the question is re-asked."""
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.REVISED, cause)

    def skip_question(self, question_id: str, cause: str) -> TransitionRecord:
        """
        REQUIRED -> SKIPPED: the participant already told us this
        information (earlier in the conversation), so the question is
        never delivered to them — but the criteria that depend on it can
        still be evaluated, because the information is known, not
        missing. The question is locked for this run (it is never asked);
        the interview continues — skipping does not end the run.
        """
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.SKIPPED, cause)

    # ------------------------------------------------------------------
    # criterion transitions (four-valued logic)
    # ------------------------------------------------------------------

    def set_criterion(
        self, criterion_id: str, status: CriterionStatus, cause: str
    ) -> TransitionRecord:
        """
        Move a criterion to `status` (OPEN -> any; UNCLEAR -> any;
        decided -> UNCLEAR only). Called by the Supervisor with the
        Diagnostiker's assessment (criterion evaluation, step 2.3).
        """
        _require_cause(cause)
        return self._apply_criterion(criterion_id, status, cause)

    # ------------------------------------------------------------------
    # run-level transitions
    # ------------------------------------------------------------------

    def complete_run(self, cause: str) -> TransitionRecord:
        """
        ACTIVE -> COMPLETED: the Navigator (via the Supervisor) decides
        there is nothing more to ask — every required question has
        reached a terminal state (CLOSED or SKIPPED). This explicit
        run-level transition (like abort_run) is the ONLY way the
        interview ends; no question-level transition ends it.
        """
        _require_cause(cause)
        return self._apply_run(RunStatus.COMPLETED, cause)

    def abort_run(self, cause: str) -> TransitionRecord:
        """ACTIVE -> ABORTED: the run ended before completion."""
        _require_cause(cause)
        return self._apply_run(RunStatus.ABORTED, cause)

    # ------------------------------------------------------------------
    # read-only access (copies — no external code can mutate the run)
    # ------------------------------------------------------------------

    @property
    def definition(self) -> InterviewDefinition:
        return self._definition

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def question_states(self) -> dict[str, QuestionState]:
        """A copy: mutating it does NOT change the run."""
        return dict(self._question_states)

    @property
    def criterion_states(self) -> dict[str, CriterionStatus]:
        """A copy: mutating it does NOT change the run."""
        return dict(self._criterion_states)

    @property
    def transitions(self) -> tuple[TransitionRecord, ...]:
        """The full transition log (in order), as a copy."""
        return tuple(self._transitions)

    def question_state(self, question_id: str) -> QuestionState:
        return self._question_states[self._known_question(question_id)]

    def criterion_state(self, criterion_id: str) -> CriterionStatus:
        return self._criterion_states[self._known_criterion(criterion_id)]

    def reserved_question(self) -> str | None:
        """The single question currently RESERVED (the next question), if any."""
        for q, s in self._question_states.items():
            if s is QuestionState.RESERVED:
                return q
        return None

    def awaiting_answer(self) -> str | None:
        """The question currently DELIVERED and waiting for the answer, if any."""
        for q, s in self._question_states.items():
            if s is QuestionState.DELIVERED:
                return q
        return None

    def revised_questions(self) -> list[str]:
        """
        All questions currently REVISED (awaiting the re-ask). The
        Navigator must NOT move on to a new question while this is
        non-empty (see the QuestionState docstring in contracts).
        """
        return [q for q, s in self._question_states.items() if s is QuestionState.REVISED]

    def has_revised_question(self) -> bool:
        return any(s is QuestionState.REVISED for s in self._question_states.values())

    def decided_criteria(self) -> dict[str, CriterionStatus]:
        """All criteria with a decided status (SATISFIED or NOT_SATISFIED)."""
        return {
            c: s
            for c, s in self._criterion_states.items()
            if s in (CriterionStatus.SATISFIED, CriterionStatus.NOT_SATISFIED)
        }

    def snapshot(self, run_version: int = 1) -> RunState:
        """
        Build a RunState snapshot of the current state (contracts.RunState
        — the type the Supervisor persists via store.py). The Supervisor
        owns run_version bookkeeping (bumped()/save_state); this is just
        the value projection.
        """
        return RunState(
            run_id=self._run_id,
            status=self._status,
            definition_version=self._definition.definition_version,
            run_version=run_version,
            question_states=dict(self._question_states),
            criterion_states=dict(self._criterion_states),
        )
