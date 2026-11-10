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
      ACTIVE  -> PAUSED | COMPLETED | ABORTED
      PAUSED  -> ACTIVE            (resume — WInf-inspired: the
                                    technical-failure pause is the
                                    PARTICIPANT's choice, never the
                                    system's; a paused run is frozen
                                    but resumable, the pending question
                                    is re-asked on resume)
      COMPLETED, ABORTED are terminal
      (the ONLY way a run ENDS — no question-level transition,
       including a skip, ends the interview; a pause is not an end)
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
                         SKIPPED          (WInf-inspired participant-
                                            initiated skip: the participant
                                            explicitly asks to skip the
                                            CURRENT question or refuses to
                                            answer it — the skip happens
                                            only once the skip-probe
                                            threshold is reached (the
                                            question is probed first: ask
                                            why + alternatives); the
                                            question is locked, and a skip
                                            is NEVER a negative answer: no
                                            readings are recorded, the
                                            criteria depending on it stay
                                            OPEN / UNCLEAR as they were)
                         REVISED          (WInf-inspired skip-probe re-ask:
                                            the participant asked to skip /
                                            refused BEFORE answering anything
                                            — the question is re-offered with
                                            the probe (ask why + alternatives)
                                            while no answer is recorded yet)
      ANSWER_RECORDED  -> CLOSED          (interpreted; no information missing)
                          REVISED         (interpreted; information missing -> re-ask)
      REVISED          -> DELIVERED       (re-entry for the re-ask; the
                                           Navigator must NOT move on to a
                                           new question while one is REVISED)
      CLOSED, SKIPPED  -> REQUIRED        (the EPISODE re-open only: a
                                           question is re-asked FOR
                                           ANOTHER EPISODE — the same
                                           question is never re-asked
                                           for the same real episode;
                                           the plain "never re-asked"
                                           rule is the special case of
                                           the once-per-(question,
                                           episode) invariant below)
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
  * a question is asked at most once PER EPISODE: (question, canonical
    episode) pairs are never repeated (require_question records the
    now-episode for the first visit; reopen_question only for a
    different episode — the episode model, design addendum 2026-10-26)
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
    EpisodeState,
    OutOfScopeExit,
    QuestionState,
    RunState,
    RunStatus,
    StateTransitionEvent,
    SubjectKind,
    ValueReading,
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
    # DELIVERED -> SKIPPED: the WInf-inspired participant-initiated skip
    # (the explicit skip command / the refusal intent — once the
    # skip-probe threshold is reached) — see the docstring table; a
    # skip is never a negative answer.
    # DELIVERED -> REVISED: the WInf-inspired skip-probe re-ask — the
    # participant asked to skip BEFORE answering anything, so the
    # question is re-offered (with the probe: ask why + alternatives)
    # while no answer is recorded yet (REVISED -> DELIVERED again).
    QuestionState.DELIVERED: frozenset(
        {
            QuestionState.ANSWER_RECORDED,
            QuestionState.SKIPPED,
            QuestionState.REVISED,
        }
    ),
    QuestionState.ANSWER_RECORDED: frozenset({QuestionState.CLOSED, QuestionState.REVISED}),
    QuestionState.REVISED: frozenset({QuestionState.DELIVERED}),
    # the episode re-open (CLOSED/SKIPPED -> REQUIRED) is the only move
    # OUT of a terminal question state — see require_question /
    # reopen_question for the guards that make it episode-specific.
    QuestionState.CLOSED: frozenset({QuestionState.REQUIRED}),
    QuestionState.SKIPPED: frozenset({QuestionState.REQUIRED}),
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

# The PAUSED state (WInf-inspired technical-failure policy): the
# PARTICIPANT's choice after the configured number of consecutive model
# failures — frozen but RESUMABLE; only COMPLETED/ABORTED are terminal.
_RUN_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.ACTIVE: frozenset(
        {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.ABORTED}
    ),
    RunStatus.PAUSED: frozenset({RunStatus.ACTIVE}),
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
    # the episode the transition is about (question require/re-open:
    # which episode the question is asked about; episode registry
    # events: the affected episode). None when not episode-related.
    episode: str | None = None

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
            episode=self.episode,
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
        # the episode-scoped runtime state (design addendum 2026-10-31,
        # OFQ-1/OFQ-2): the episode the CURRENT readings / criterion
        # decisions belong to, the per-episode navigation-value readings,
        # and the per-episode criterion states of episode-scoped
        # criteria (run-scoped criteria keep the single global map above).
        self._active_episode: str = "now"
        self._readings: dict[str, dict[str, ValueReading]] = {}
        self._episode_criterion_states: dict[str, dict[str, CriterionStatus]] = {}
        # set exactly once via abort_out_of_scope (OFQ-6).
        self._out_of_scope: OutOfScopeExit | None = None
        self._seed_episodes()

    # ------------------------------------------------------------------
    # the episode registry (the per-interview `episodes` block, design
    # addendum 2026-10-26)
    # ------------------------------------------------------------------

    def _seed_episodes(self) -> None:
        """
        Seed the episode registry from the definition's `episodes`
        block. ALWAYS the now-episode (id "now") — the episode every
        question is first asked about; with `past.declaration ==
        "predeclared"`, the predeclared episodes are seeded by name.
        In dynamic mode past episodes are created at RUNTIME
        (declare_episode: ep_1, ep_2, ...).
        """
        episodes = self._definition.episodes
        self._episodes: dict[str, EpisodeState] = {
            "now": EpisodeState(
                id="now",
                kind="now",
                window_days=episodes.now.window_days,
                rule=episodes.now.rule,
            )
        }
        self._episode_order: list[str] = ["now"]
        for name in episodes.past.episodes:
            self._episodes[name] = EpisodeState(id=name, kind="past", label=name)
            self._episode_order.append(name)
        self._episode_counter = 0
        self._question_episodes: dict[str, list[str]] = {}

    @property
    def episodes(self) -> dict[str, EpisodeState]:
        """A copy of the episode registry (id -> EpisodeState)."""
        return dict(self._episodes)

    @property
    def question_episodes(self) -> dict[str, list[str]]:
        """
        Which episodes each question was asked about (in order; a copy
        of the lists). The first visit is the now-episode.
        """
        return {q: list(eps) for q, eps in self._question_episodes.items()}

    def canonical_episode(self, episode_id: str) -> str:
        """
        The canonical id of an episode (follows a merge one hop: the
        winner of a merge is never merged again). Fails loudly on an
        unknown id.
        """
        episode = self._episodes.get(episode_id)
        if episode is None:
            raise ValueError(
                f"unknown episode id {episode_id!r} (known: "
                f"{', '.join(self._episodes) or 'none'})"
            )
        return episode.merged_into or episode_id

    def set_active_episode(self, episode_id: str, cause: str) -> TransitionRecord:
        """
        Make `episode_id` the ACTIVE episode context (design addendum
        2026-10-31, OFQ-1): the episode the CURRENT navigation-value
        readings and the criterion decisions of episode-scoped criteria
        belong to. The episode must exist in the registry (declare it
        first in dynamic mode). Called by the Supervisor when it
        re-enters the interview for a past episode — the re-entry itself
        (re-opening the target questions) is the Supervisor's job, this
        only switches the context.
        """
        _require_cause(cause)
        self._require_active()
        target = self.canonical_episode(episode_id)
        current = self._active_episode
        self._active_episode = target
        return self._log(
            SubjectKind.EPISODE,
            target,
            f"active:{current}",
            f"active:{target}",
            cause,
        )

    def _require_dynamic_episodes(self, operation: str) -> None:
        """Past-episode registry operations need enabled + dynamic mode."""
        episodes = self._definition.episodes
        if not episodes.enabled:
            raise IllegalTransitionError(
                SubjectKind.EPISODE,
                "",
                "none",
                operation,
                "the definition's episodes block is disabled — no episode "
                "machinery exists for this interview",
            )
        if episodes.past.declaration != "dynamic":
            raise IllegalTransitionError(
                SubjectKind.EPISODE,
                "",
                "none",
                operation,
                f"past episodes are {episodes.past.declaration!r} — "
                "dynamic-mode operations (declare/alias/merge) apply only "
                "with past.declaration 'dynamic'",
            )

    def _note_question_episode(self, question_id: str, episode: str) -> None:
        """Record that the question was asked about the (canonical) episode."""
        self._question_episodes.setdefault(question_id, []).append(episode)

    def declare_episode(self, label: str, cause: str) -> TransitionRecord:
        """
        Declare a NEW past episode (dynamic mode only): the clinician
        declares "another episode", after which the patient specifies
        it in their own words (record_episode_alias). The run assigns
        the deterministic id ep_1, ep_2, ... The returned record's
        subject_id is the new episode id.
        """
        _require_cause(cause)
        self._require_active()
        self._require_dynamic_episodes("declare_episode")
        self._episode_counter += 1
        episode_id = f"ep_{self._episode_counter}"
        self._episodes[episode_id] = EpisodeState(
            id=episode_id, kind="past", label=label.strip() or None
        )
        self._episode_order.append(episode_id)
        return self._log(SubjectKind.EPISODE, episode_id, "none", "declared", cause)

    def record_episode_alias(
        self, episode_id: str, alias: str, cause: str
    ) -> TransitionRecord:
        """
        Record one of the patient's VERBATIM framings of an episode
        (dynamic mode only): "march 2026", "the month before I started
        uni", ... Aliases accumulate on the CANONICAL episode (a
        merged episode's aliases live on its winner) and are the
        reference material for the Diagnostiker's identity judgments.
        """
        _require_cause(cause)
        self._require_active()
        self._require_dynamic_episodes("record_episode_alias")
        canonical = self.canonical_episode(episode_id)
        episode = self._episodes[canonical]
        if alias not in episode.aliases:
            self._episodes[canonical] = episode.model_copy(
                update={"aliases": episode.aliases + (alias,)}
            )
        return self._log(
            SubjectKind.EPISODE, canonical, "declared", "aliased", cause
        )

    def merge_episodes(
        self, episode_id_a: str, episode_id_b: str, cause: str
    ) -> TransitionRecord:
        """
        Merge two episode declarations into ONE real episode (dynamic
        mode only): the Diagnostiker judged them identical, the run
        unifies them — the EARLIER episode (creation order) is the
        canonical one, the later one becomes an alias of it
        (merged_into, aliases merged into the winner). The returned
        record's subject_id is the merged-away (non-canonical) id.
        """
        _require_cause(cause)
        self._require_active()
        self._require_dynamic_episodes("merge_episodes")
        a = self.canonical_episode(episode_id_a)
        b = self.canonical_episode(episode_id_b)
        if a == b:
            raise IllegalTransitionError(
                SubjectKind.EPISODE,
                episode_id_b,
                "merged",
                f"canonical:{a}",
                f"{episode_id_a!r} and {episode_id_b!r} are already the "
                f"same episode ({a!r}) — nothing to merge",
            )
        winner, loser = sorted((a, b), key=self._episode_order.index)
        self._episodes[loser] = self._episodes[loser].model_copy(
            update={"merged_into": winner}
        )
        merged_aliases = self._episodes[winner].aliases + tuple(
            alias
            for alias in self._episodes[loser].aliases
            if alias not in self._episodes[winner].aliases
        )
        self._episodes[winner] = self._episodes[winner].model_copy(
            update={"aliases": merged_aliases}
        )
        return self._log(
            SubjectKind.EPISODE,
            loser,
            "standalone",
            f"merged_into:{winner}",
            cause,
        )

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
        episode: str | None = None,
    ) -> TransitionRecord:
        record = TransitionRecord(
            subject=subject,
            subject_id=subject_id,
            from_state=from_state,
            to_state=to_state,
            cause=cause,
            timestamp=utc_now(),
            episode=episode,
        )
        self._transitions.append(record)
        return record

    def _apply_question(
        self,
        question_id: str,
        target: QuestionState,
        cause: str,
        episode: str | None = None,
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
        return self._log(
            SubjectKind.QUESTION, question_id, current.value, target.value, cause,
            episode=episode,
        )

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

    def require_question(
        self, question_id: str, cause: str, episode_id: str = "now"
    ) -> TransitionRecord:
        """
        UNSEEN -> REQUIRED: the Navigator marks the question as
        currently required — its FIRST visit. The question is asked
        about the given episode (default: the now-episode) and the
        (question, episode) pair is recorded (the audit trail + the
        once-per-(question, episode) invariant).
        """
        _require_cause(cause)
        question_id = self._known_question(question_id)
        current = self._question_states[question_id]
        if current is not QuestionState.UNSEEN:
            raise IllegalTransitionError(
                SubjectKind.QUESTION,
                question_id,
                current.value,
                QuestionState.REQUIRED.value,
                "require is only for UNSEEN questions; a CLOSED/SKIPPED "
                "question is re-asked via reopen_question (another episode)",
            )
        episode = self.canonical_episode(episode_id)
        record = self._apply_question(
            question_id, QuestionState.REQUIRED, cause, episode=episode
        )
        self._note_question_episode(question_id, episode)
        return record

    def reopen_question(
        self, question_id: str, episode_id: str, cause: str
    ) -> TransitionRecord:
        """
        CLOSED/SKIPPED -> REQUIRED: re-ask the question FOR ANOTHER
        EPISODE (the episode model, design addendum 2026-10-26). The
        question is never re-asked for the same REAL episode: if the
        (canonical) episode was already asked for this question, this
        is an IllegalTransitionError. Called by the Supervisor when it
        applies the Navigator's reopen_proposals (an episode jump
        re-entering an already-assessed target group).
        """
        _require_cause(cause)
        question_id = self._known_question(question_id)
        self._require_active()
        current = self._question_states[question_id]
        if current not in (QuestionState.CLOSED, QuestionState.SKIPPED):
            raise IllegalTransitionError(
                SubjectKind.QUESTION,
                question_id,
                current.value,
                QuestionState.REQUIRED.value,
                "reopen is only for CLOSED/SKIPPED questions (a UNSEEN "
                "question is required, not re-opened)",
            )
        episode = self.canonical_episode(episode_id)
        asked = self._question_episodes.get(question_id, ())
        if episode in asked:
            raise IllegalTransitionError(
                SubjectKind.QUESTION,
                question_id,
                current.value,
                QuestionState.REQUIRED.value,
                f"the question was already asked for episode {episode!r} "
                f"(asked for: {', '.join(asked)}) — the same question is "
                "never re-asked for the same episode",
            )
        record = self._apply_question(
            question_id, QuestionState.REQUIRED, cause, episode=episode
        )
        self._note_question_episode(question_id, episode)
        return record

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
        """
        ANSWER_RECORDED -> REVISED: the answer lacks information, the
        question is re-asked. Also DELIVERED -> REVISED (WInf-inspired
        skip-probe re-ask): the participant asked to skip / refused
        BEFORE answering anything, so the question is re-offered with
        the probe (ask why + alternatives) while no answer is recorded
        yet (REVISED -> DELIVERED again).
        """
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

        Also DELIVERED -> SKIPPED (WInf-inspired participant-initiated
        skip): the participant explicitly asked to skip the CURRENT
        question or refused to answer it — the skip happens only once
        the skip-probe threshold is reached (the question is probed
        first: ask why + alternatives, a vague/short answer is fine).
        Same locking semantics, one deliberate difference: the
        information is NOT known, so a skip is NEVER a negative answer
        — no readings are recorded and the criteria depending on it
        stay OPEN/UNCLEAR as they were (the cause carries the skip
        reason: explicit / refused / unresolved).
        """
        _require_cause(cause)
        return self._apply_question(question_id, QuestionState.SKIPPED, cause)

    # ------------------------------------------------------------------
    # navigation values — the per-EPISODE readings (design addendum
    # 2026-10-31, OFQ-2): the criterion engine evaluates a criterion
    # only against the readings of the context its scope selects, so a
    # past episode's values cannot contaminate the now-episode's
    # decisions. A reading is NOT a state transition (there is no
    # value subject in the audit vocabulary): it is the current content
    # of a value in one episode's context, and storing it is a store
    # operation, not a move in the transition tables.
    # ------------------------------------------------------------------

    def set_value(self, reading: ValueReading, cause: str) -> None:
        """Store one value's reading in the ACTIVE episode's context."""
        self.set_value_for(self._active_episode, reading, cause)

    def set_value_for(
        self, episode_id: str, reading: ValueReading, cause: str
    ) -> None:
        """
        Store one value's reading in the given episode's context
        (canonical id; must exist in the registry). Readings overwrite:
        the LATEST reading of a value in a context wins — exactly the
        semantics the criteria engine assumes for a current reading.
        """
        _require_cause(cause)
        self._require_active()
        context = self.canonical_episode(episode_id)
        self._readings.setdefault(context, {})[reading.name] = reading

    def readings(self, episode_id: str | None = None) -> dict[str, ValueReading]:
        """
        A copy of one episode context's current readings (value name ->
        ValueReading). Default: the ACTIVE context.
        """
        context = (
            self._active_episode
            if episode_id is None
            else self.canonical_episode(episode_id)
        )
        return dict(self._readings.get(context, {}))

    def readings_by_context(self) -> dict[str, dict[str, ValueReading]]:
        """A copy of ALL per-context readings (context id -> readings)."""
        return {
            context: dict(readings)
            for context, readings in self._readings.items()
        }

    def all_readings(self) -> dict[str, ValueReading]:
        """
        The RUN-SCOPED view: all contexts merged (later episode context
        overrides an earlier one — the legacy single-mapping behavior,
        what unscoped criteria evaluate against).
        """
        merged: dict[str, ValueReading] = {}
        ordered = [
            context
            for context in self._episode_order
            if context in self._readings
        ] + [
            context
            for context in self._readings
            if context not in self._episode_order
        ]
        for context in ordered:
            merged.update(self._readings[context])
        return merged

    def value(
        self, name: str, episode_id: str | None = None
    ) -> ValueReading | None:
        """The current reading of one value (default: the active context)."""
        return self.readings(episode_id).get(name)

    # ------------------------------------------------------------------
    # criterion transitions (four-valued logic)
    # ------------------------------------------------------------------

    def _criterion_scope(self, criterion_id: str):
        """The criterion's scope (None: run-scoped — the legacy mode)."""
        for criterion in self._definition.all_criteria():
            if criterion.id == criterion_id:
                return criterion.scope
        return None

    def _apply_episode_criterion(
        self,
        context: str,
        criterion_id: str,
        target: CriterionStatus,
        cause: str,
    ) -> TransitionRecord:
        """The same transition-table check as _apply_criterion, per context."""
        self._require_active()
        self._known_criterion(criterion_id)
        states = self._episode_criterion_states.setdefault(context, {})
        current = states.get(criterion_id, CriterionStatus.OPEN)
        legal = _CRITERION_TRANSITIONS[current]
        if target not in legal:
            legal_names = ", ".join(s.value for s in legal) or "none (terminal state)"
            raise IllegalTransitionError(
                SubjectKind.CRITERION,
                f"{criterion_id} (episode {context})",
                current.value,
                target.value,
                f"legal from {current.value}: {legal_names}",
            )
        states[criterion_id] = target
        return self._log(
            SubjectKind.CRITERION,
            criterion_id,
            current.value,
            target.value,
            cause,
            episode=context,
        )

    def set_criterion(
        self, criterion_id: str, status: CriterionStatus, cause: str
    ) -> TransitionRecord:
        """
        Move a criterion to `status` (OPEN -> any; UNCLEAR -> any;
        decided -> UNCLEAR only). Called by the Supervisor with the
        Diagnostiker's assessment (criterion evaluation, step 2.3).

        Episode-scoped criteria (OFQ-2) are decided IN THE ACTIVE
        EPISODE CONTEXT — the Supervisor switches the context before
        evaluating a past episode, so the decision lands in the right
        map; a `past`-scoped criterion cannot be decided in the
        now-episode context at all (its values are not asked there).
        """
        _require_cause(cause)
        self._known_criterion(criterion_id)
        scope = self._criterion_scope(criterion_id)
        if scope is None:
            return self._apply_criterion(criterion_id, status, cause)
        context = self._active_episode
        if scope.episode == "past" and self._episodes[context].kind != "past":
            raise IllegalTransitionError(
                SubjectKind.CRITERION,
                criterion_id,
                "past-scoped",
                status.value,
                "a past-scoped criterion can only be decided in a "
                "past-episode context (the active context is "
                f"'{context}')",
            )
        return self._apply_episode_criterion(context, criterion_id, status, cause)

    def force_criterion(
        self, criterion_id: str, status: CriterionStatus, cause: str
    ) -> list[TransitionRecord]:
        """
        Force-code a criterion to a decided status (design addendum
        2026-10-31): the SCID escalation instruction (A52: "code A30
        with '+' and A38 with 'JA'") is a CODING decision, not a
        derived one — no answer of the active episode produced it.
        The transition table forbids a direct decided -> decided move,
        so an already-decided criterion is reopened via UNCLEAR first.
        Returns the applied records (empty when the criterion already
        holds `status` in the active context — the no-op case).
        """
        _require_cause(cause)
        self._known_criterion(criterion_id)
        current = self.criterion_state(criterion_id)
        if current is status:
            return []
        records: list[TransitionRecord] = []
        if current in (CriterionStatus.SATISFIED, CriterionStatus.NOT_SATISFIED):
            records.append(
                self.set_criterion(
                    criterion_id,
                    CriterionStatus.UNCLEAR,
                    f"{cause} (reopened)",
                )
            )
        records.append(self.set_criterion(criterion_id, status, cause))
        return records

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

    def pause_run(self, cause: str) -> TransitionRecord:
        """
        ACTIVE -> PAUSED (WInf-inspired technical-failure policy):
        the PARTICIPANT's explicit choice after the configured number
        of consecutive technical (model) failures. The run is frozen
        (no question/criterion changes while PAUSED) but NOT ended —
        it can be resumed (resume_run). The Supervisor delivers the
        pre-approved pause confirmation and persists the snapshot.
        """
        _require_cause(cause)
        return self._apply_run(RunStatus.PAUSED, cause)

    def resume_run(self, cause: str) -> TransitionRecord:
        """
        PAUSED -> ACTIVE: the participant resumes the paused interview.
        The pending question (still DELIVERED — its answer was never
        recorded) is re-asked by the Supervisor (a pure output-layer
        retransmission, no question-state transition).
        """
        _require_cause(cause)
        return self._apply_run(RunStatus.ACTIVE, cause)

    def abort_out_of_scope(
        self, target_ref: str, note: str = "", cause: str | None = None
    ) -> TransitionRecord:
        """
        End the run via the OUT-OF-SCOPE exit (design addendum
        2026-10-31, OFQ-6): the interview's routed continuation target
        lies OUTSIDE this definition (the SCID excerpt ends at A78).
        The OutOfScopeExit is recorded on the run (surfaced by the
        snapshot / audit) and the run is aborted — an explicit,
        logged hand-over, never a silently swallowed jump.
        """
        self._require_active()
        self._out_of_scope = OutOfScopeExit(
            target_ref=target_ref, note=note.strip() or None
        )
        if cause is None:
            cause = (
                "out-of-scope exit: the interview continues with "
                f"{target_ref} (outside this definition)"
            )
        return self.abort_run(cause)

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
        """
        The criterion's status in the ACTIVE episode context (the
        episode-scoped model, design addendum 2026-10-31, OFQ-2):
        run-scoped criteria keep the single global state;
        episode-scoped criteria are read from the state map of the
        active episode (a criterion that has not been decided in this
        context is OPEN there, whatever other contexts say).
        """
        criterion_id = self._known_criterion(criterion_id)
        scope = self._criterion_scope(criterion_id)
        if scope is None:
            return self._criterion_states[criterion_id]
        context = self._active_episode
        return self._episode_criterion_states.get(context, {}).get(
            criterion_id, CriterionStatus.OPEN
        )

    @property
    def active_episode(self) -> str:
        """The episode the current readings / criterion decisions belong to."""
        return self._active_episode

    @property
    def out_of_scope(self) -> OutOfScopeExit | None:
        """The out-of-scope exit (set once via abort_out_of_scope)."""
        return self._out_of_scope

    def episode_criterion_states(
        self, episode_id: str | None = None
    ) -> dict[str, CriterionStatus]:
        """
        A copy of one episode context's criterion states (criterion id
        -> status; default: the ACTIVE context). Run-scoped criteria are
        NOT in these maps (they live in criterion_states).
        """
        context = (
            self._active_episode
            if episode_id is None
            else self.canonical_episode(episode_id)
        )
        return dict(self._episode_criterion_states.get(context, {}))

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
            question_episodes={
                q: list(eps) for q, eps in self._question_episodes.items()
            },
            episodes=dict(self._episodes),
            active_episode=self._active_episode,
            out_of_scope=self._out_of_scope,
        )
