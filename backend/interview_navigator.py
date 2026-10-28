"""
INTERVIEW NAVIGATOR — THE DETERMINISTIC NAVIGATION TOOL (PHASE 2, 2.4)
=======================================================================
The "Interview Navi" (docs/interview_navi_design.md — the binding design
reference): a DETERMINISTIC TOOL (not an LLM agent) that is the
authoritative source for the interview's navigation. It does not
diagnose, formulate participant-facing content, or mutate anything: it
computes navigation decisions over (InterviewDefinition, InterviewRun,
value readings) and returns them as proposals. The Supervisor (step
2.5) is the sole authority that applies them to the run (the only
mutation path, step 2.2) and writes the audit trail.

Responsibilities (design note, "Bezug zum Interviewer-Agenten"):
  * maintain the "questions to be asked" set — the still-open questions
    of the run (outstanding_questions / open_questions),
  * hand the interviewer agent a CANDIDATE SET (contracts.
    QuestionCandidateSet) to pick the most sensible next question
    from — ordered by the deterministic fallback order with a per-
    candidate reason,
  * the DETERMINISTIC FALLBACK ORDER — the postorder traversal of the
    hierarchy: nested sections first, then question groups in listed
    order, then questions in their defined order; when no criteria
    decide otherwise, that order IS the question order,
  * resolve JUMPS after a question's answer is recorded and
    interpreted: a jump fires when the gated criterion's state in the
    run matches the jump's `when` (a jump without on_criterion fires
    unconditionally); the FIRST declared jump that fires wins; after a
    jump, navigation follows the postorder traversal from the target;
    questions bypassed by the jump receive SKIPPED proposals
    ("not needed in the current flow"); a question is never re-asked
    in the same run (a jump target that is fully answered — "locked"
    — is handled by the loop policy below),
  * evaluate criteria groups ONLINE (design note: criteria are
    evaluated continuously, as soon as the answers that feed them are
    complete): propose_criterion_updates evaluates every criterion
    after each answer — the criteria whose values are present evaluate
    to a decided status, the rest stay OPEN/UNCLEAR — and applies the
    DECIDED GUARD: a criterion already SATISFIED/NOT_SATISFIED is not
    re-evaluated (reopening only through UNCLEAR, step 2.2's
    criterion transition table).

OPEN POINT — loop-back to an already-answered (LOCKED) group,
deliberately left open by the design note (RECONSTRUCTION_PLAN.md §5,
open question 6): a jump whose target subtree has no open questions
(no more question to ask there). The LoopPolicy decides what the
Navigator does in that case:
  * IGNORE_AND_CONTINUE (DEFAULT, safe): the jump is ignored and the
    deterministic postorder continues from the current position — no
    locked question is ever re-asked, nothing is skipped;
  * END_RUN: the loop is treated as "nothing more to ask" — every
    remaining open question receives a SKIPPED proposal and the next
    question is None (the Supervisor completes the run);
  * ERROR: NavigatorError — the situation is not silently navigated.
The default is a PRAGMATIC CHOICE, not the final decision: the policy
is a constructor argument, so the final ruling can be slotted in
without changing the navigation logic.

Used by: the Supervisor (step 2.5) and the tests (step 2.6). The
legacy pipeline does not use this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from contracts import (
    CriterionStatus,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionState,
    RunStatus,
    ValueReading,
)
from criteria_engine import evaluate_criterion
from interview_definition import (
    InterviewDefinition,
    Jump,
    QuestionGroup,
    SectionGroup,
)
from interview_run import InterviewRun


class LoopPolicy(Enum):
    """
    What the Navigator does when a jump leads to an already-answered
    (locked) group (see the module docstring). The default
    IGNORE_AND_CONTINUE is a pragmatic choice pending the open decision
    (RECONSTRUCTION_PLAN.md §5, question 6).
    """

    IGNORE_AND_CONTINUE = "ignore_and_continue"
    END_RUN = "end_run"
    ERROR = "error"


class NavigatorError(RuntimeError):
    """
    A navigation situation the Navigator cannot resolve (the ERROR loop
    policy, or an unknown jump target). Fails loudly instead of
    guessing.
    """


@dataclass(frozen=True)
class JumpResolution:
    """
    The outcome of resolving the jumps declared on one answered
    question. A PROPOSAL — the Supervisor applies it:
      * skip_question() for every id in skip_proposals (UNSEEN ->
        REQUIRED -> SKIPPED; "not needed in the current flow"),
      * then require_question()/reserve_question() for next_question
        (when not None),
      * when policy_applied is END_RUN and next_question is None:
        skip the proposals and complete_run().
    """

    fired: bool
    jump: Jump | None
    target_id: str | None
    next_question: str | None
    skip_proposals: tuple[str, ...]
    locked_target: bool
    policy_applied: LoopPolicy | None


# The question states of the state machine (step 2.2) this tool reads:
_TERMINAL_STATES = frozenset({QuestionState.CLOSED, QuestionState.SKIPPED})  # answered or never-to-be-delivered
_CANDIDATE_STATES = frozenset({QuestionState.UNSEEN, QuestionState.REQUIRED})  # can be asked (not yet in flight, not yet decided)


class InterviewNavigator:
    """
    The deterministic navigation tool for ONE interview definition.

    Pure and stateless per call: every method is a function of the
    (definition, run, readings) it is given — the same inputs always
    give the same proposals. The navigator never mutates the run; the
    Supervisor applies the proposals through the state machine's
    transition methods (the only mutation path).
    """

    def __init__(
        self,
        definition: InterviewDefinition,
        loop_policy: LoopPolicy = LoopPolicy.IGNORE_AND_CONTINUE,
    ) -> None:
        self._definition = definition
        self._loop_policy = loop_policy
        # the deterministic fallback order: the postorder traversal of
        # the hierarchy — nested sections first, then question groups
        # in listed order, then questions in their defined order.
        self._order: list[str] = [q.id for q in definition.all_questions()]
        self._position: dict[str, int] = {
            qid: i for i, qid in enumerate(self._order)
        }
        self._declarations = {v.name: v for v in definition.values}
        self._question_group_of: dict[str, QuestionGroup] = {
            q.id: group for group, q in definition.questions_with_group()
        }

    # ------------------------------------------------------------------
    # structure access (the authoritative source for interview structure)
    # ------------------------------------------------------------------

    @property
    def definition(self) -> InterviewDefinition:
        return self._definition

    @property
    def loop_policy(self) -> LoopPolicy:
        return self._loop_policy

    @property
    def question_order(self) -> list[str]:
        """
        The deterministic fallback order (postorder / document order)
        of all single questions — QuestionCandidateSet.fallback_order.
        """
        return list(self._order)

    def _section_questions(self, section: SectionGroup) -> list[str]:
        """All questions in a section group's subtree, document order."""
        out: list[str] = []
        for child in section.children:
            if isinstance(child, SectionGroup):
                out.extend(self._section_questions(child))
            else:
                out.extend(q.id for q in child.questions)
        return out

    def _questions_of_target(self, target_id: str) -> list[str]:
        """
        The questions of a jump target's subtree (document order). The
        target is a question, question group or section group (import
        validation guarantees it exists — fail loudly if not).
        """
        question = self._definition.question_by_id(target_id)
        if question is not None:
            return [target_id]
        group = self._definition.group_by_id(target_id)
        if group is not None:
            return [q.id for q in group.questions]
        section = self._definition.section_by_id(target_id)
        if section is not None:
            return self._section_questions(section)
        raise NavigatorError(
            f"jump target {target_id!r} is not a known question, group or "
            f"section id (import-time validation would have rejected it)"
        )

    # ------------------------------------------------------------------
    # the "questions to be asked" set 
    # ------------------------------------------------------------------

    def outstanding_questions(self, run: InterviewRun) -> list[str]:
        """
        The still-open questions — everything that is not CLOSED or
        SKIPPED yet (UNSEEN .. REVISED), in postorder. The run ends
        only when this set is empty (is_complete).
        """
        states = run.question_states
        return [q for q in self._order if states[q] not in _TERMINAL_STATES]

    def open_questions(self, run: InterviewRun) -> list[str]:
        """
        The questions that can be ASKED right now — UNSEEN or REQUIRED
        (not yet in flight: RESERVED / DELIVERED / ANSWER_RECORDED /
        REVISED), in postorder.
        """
        states = run.question_states
        return [q for q in self._order if states[q] in _CANDIDATE_STATES]

    def is_complete(self, run: InterviewRun) -> bool:
        """
        Nothing more is to be asked: every question is CLOSED or
        SKIPPED. The Supervisor may then complete_run() (the ONLY way
        a run ends, step 2.2).
        """
        states = run.question_states
        return all(states[q] in _TERMINAL_STATES for q in self._order)

    # ------------------------------------------------------------------
    # the candidate set handed to the interviewer agent
    # ------------------------------------------------------------------

    def _candidate_reason(self, question_id: str) -> str:
        """The human-readable (auditable) reason for one candidate."""
        question = self._definition.question_by_id(question_id)
        group = self._question_group_of[question_id]
        position = self._position[question_id] + 1
        required = "required" if question.required else "optional"
        return (
            f"postorder position {position} in group {group.id} "
            f"({group.strategy}), {required}"
        )

    def _candidate_list(self, run: InterviewRun) -> list[QuestionCandidate]:
        """
        The candidates for the interviewer agent (the set's body).

        Rules (in priority order):
          * a terminal run (COMPLETED / ABORTED) has no candidates;
          * while a question is REVISED it is re-asked FIRST — the
            Navigator must not move on to a new question (the REVISED
            questions come first; a RESERVED question, if any,
            follows) — no new candidates;
          * while a question is RESERVED it is the single next
            question (the reservation was made, it is in flight);
          * while a question is DELIVERED or ANSWER_RECORDED the run
            waits for the answer — no candidates;
          * otherwise: every UNSEEN/REQUIRED question is a candidate,
            in the deterministic fallback order (postorder); ordering
            0 is the fallback next question. The interviewer agent may
            pick any candidate as "the most sensible next question";
            the Supervisor reserves the picked one.
        """
        states = run.question_states
        if run.status is not RunStatus.ACTIVE:
            return []
        revised = [q for q in self._order if states[q] is QuestionState.REVISED]
        reserved = run.reserved_question()
        if revised:
            return self._revised_candidates(revised, reserved)
        if reserved is not None:
            return self._reserved_candidates(reserved)
        in_flight = [
            q
            for q in self._order
            if states[q]
            in (QuestionState.DELIVERED, QuestionState.ANSWER_RECORDED)
        ]
        if in_flight:
            return []  # the answer is in flight; nothing to ask right now
        return self._open_candidates(run)

    def _revised_candidates(
        self, revised: list[str], reserved: str | None
    ) -> list[QuestionCandidate]:
        """REVISED questions are re-asked before any new question."""
        candidates = [
            QuestionCandidate(
                question_id=q,
                text=self._definition.question_by_id(q).text,
                group_id=self._question_group_of[q].id,
                reason_required=(
                    "REVISED: re-ask before any new question "
                    f"({self._candidate_reason(q)})"
                ),
                ordering=i,
            )
            for i, q in enumerate(revised)
        ]
        if reserved is not None:
            candidates.extend(self._reserved_candidates(reserved))
        return candidates

    def _reserved_candidates(self, reserved: str) -> list[QuestionCandidate]:
        """The single RESERVED question, as the only candidate."""
        return [
            QuestionCandidate(
                question_id=reserved,
                text=self._definition.question_by_id(reserved).text,
                group_id=self._question_group_of[reserved].id,
                reason_required=(
                    "RESERVED: the single next question "
                    f"({self._candidate_reason(reserved)})"
                ),
                ordering=0,
            )
        ]

    def _open_candidates(self, run: InterviewRun) -> list[QuestionCandidate]:
        """Every open (UNSEEN/REQUIRED) question, deterministic order."""
        return [
            QuestionCandidate(
                question_id=q,
                text=self._definition.question_by_id(q).text,
                group_id=self._question_group_of[q].id,
                reason_required=self._candidate_reason(q),
                ordering=i,
            )
            for i, q in enumerate(self.open_questions(run))
        ]

    def candidate_set(self, run: InterviewRun, turn_id: str) -> QuestionCandidateSet:
        """
        The candidate set handed to the interviewer agent (contracts.
        QuestionCandidateSet, identified by run_id + turn_id like every
        other per-turn artifact): which questions may be asked next, in
        the deterministic fallback order, with a per-candidate reason.
        """
        return QuestionCandidateSet(
            run_id=run.run_id,
            turn_id=turn_id,
            candidates=self._candidate_list(run),
        )

    def next_question(self, run: InterviewRun) -> str | None:
        """
        The deterministic fallback next question (ordering 0 of the
        candidate list), or None when there is nothing to ask right
        now — either the run is complete (is_complete) or a question
        is in flight awaiting its answer (DELIVERED / ANSWER_RECORDED)
        or its re-ask.
        """
        candidates = self._candidate_list(run)
        if not candidates:
            return None
        return candidates[0].question_id

    # ------------------------------------------------------------------
    # jumps (transitions out of an answered question)
    # ------------------------------------------------------------------

    def _fallback_next(self, run: InterviewRun) -> str | None:
        """
        The fallback next question when a jump does not fire (or is
        ignored by the loop policy): the first open question in
        postorder. The source question of a resolve_jump call is still
        in flight (ANSWER_RECORDED) and is therefore NOT an open
        question — so this is exactly the question that next_question
        would return once the Supervisor has closed the source.
        """
        open_questions = self.open_questions(run)
        if not open_questions:
            return None
        return open_questions[0]

    def resolve_jump(
        self, run: InterviewRun, question_id: str
    ) -> JumpResolution:
        """
        Resolve the jumps declared on `question_id` AFTER its answer
        was recorded and interpreted (the run must be ACTIVE). A jump
        fires when the state of its gated criterion (on_criterion) in
        the run matches the jump's `when`; a jump without on_criterion
        fires unconditionally. The FIRST declared jump that fires wins
        (deterministic). When no jump fires, next_question is the
        plain deterministic fallback (postorder) and nothing is
        skipped.
        """
        question = self._definition.question_by_id(question_id)
        if run.status is not RunStatus.ACTIVE:
            return JumpResolution(
                fired=False,
                jump=None,
                target_id=None,
                next_question=None,
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
            )
        if question is None or not question.jumps:
            return JumpResolution(
                fired=False,
                jump=None,
                target_id=None,
                next_question=self._fallback_next(run),
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
            )
        for jump in question.jumps:
            if jump.on_criterion is None:
                fires = True
            else:
                fires = run.criterion_state(jump.on_criterion) is jump.when
            if fires:
                return self._resolve_fired_jump(run, question_id, jump)
        return JumpResolution(
            fired=False,
            jump=None,
            target_id=None,
            next_question=self._fallback_next(run),
            skip_proposals=(),
            locked_target=False,
            policy_applied=None,
        )

    def _resolve_fired_jump(
        self, run: InterviewRun, source_id: str, jump: Jump
    ) -> JumpResolution:
        """The outcome of one FIRED jump (target resolution + skips)."""
        target_questions = self._questions_of_target(jump.to)
        states = run.question_states
        candidates = [
            q for q in target_questions if states[q] in _CANDIDATE_STATES
        ]
        if not candidates:
            # the target is LOCKED — every question of its subtree is
            # terminal (answered or skipped). Design-note open point;
            # the loop policy decides (see the module docstring).
            return self._apply_loop_policy(run, source_id, jump)
        next_question = candidates[0]
        source_pos = self._position[source_id]
        target_start = self._position[target_questions[0]]
        if target_start > source_pos:
            # FORWARD jump: the open questions strictly between the
            # source and the target subtree are bypassed by the jump
            # ("not needed in the current flow") -> SKIPPED proposals.
            skip_proposals = tuple(
                q
                for q in self.open_questions(run)
                if source_pos < self._position[q] < target_start
            )
        else:
            # BACKWARD jump: the flow re-enters the target subtree and
            # continues the postorder traversal from there — nothing
            # is bypassed permanently.
            skip_proposals = ()
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=next_question,
            skip_proposals=skip_proposals,
            locked_target=False,
            policy_applied=None,
        )

    def _apply_loop_policy(
        self, run: InterviewRun, source_id: str, jump: Jump
    ) -> JumpResolution:
        """
        A fired jump whose target is LOCKED (open question 6, §5).
        The locked questions are never re-asked under any policy.
        """
        if self._loop_policy is LoopPolicy.ERROR:
            raise NavigatorError(
                f"jump of {source_id!r} leads to the already answered "
                f"(locked) target {jump.to!r}; the loop-back behavior is "
                f"undecided (RECONSTRUCTION_PLAN.md, open question 6) "
                f"and the navigator is configured to fail loudly"
            )
        if self._loop_policy is LoopPolicy.END_RUN:
            # the loop counts as "nothing more to ask": everything
            # remaining is skipped, the run may be completed
            return JumpResolution(
                fired=True,
                jump=jump,
                target_id=jump.to,
                next_question=None,
                skip_proposals=tuple(self.open_questions(run)),
                locked_target=True,
                policy_applied=LoopPolicy.END_RUN,
            )
        # DEFAULT — IGNORE_AND_CONTINUE: the jump is ignored, the
        # deterministic postorder continues from the current position.
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=self._fallback_next(run),
            skip_proposals=(),
            locked_target=True,
            policy_applied=LoopPolicy.IGNORE_AND_CONTINUE,
        )

    # ------------------------------------------------------------------
    # online criteria evaluation (the criteria groups of the design note)
    # ------------------------------------------------------------------

    def propose_criterion_updates(
        self,
        run: InterviewRun,
        readings: dict[str, ValueReading],
    ) -> dict[str, CriterionStatus]:
        """
        The ONLINE criteria evaluation (design note: criteria are
        evaluated continuously, as soon as the answers that feed them
        are complete). Every criterion is evaluated NOW — the engine
        returns OPEN/UNCLEAR for criteria whose values are not (fully)
        present yet — and exactly the status changes that differ from
        the current state are returned as proposals.

        The DECIDED GUARD (plan §1.1 rule 4): a criterion already
        SATISFIED or NOT_SATISFIED is NOT re-evaluated — its status
        only changes through UNCLEAR (reopened when a clarification
        shows the earlier answer was misread), never decided ->
        decided. That is exactly the state machine's criterion
        transition table (step 2.2); every proposal returned here is a
        legal transition (the current state is OPEN or UNCLEAR).

        The Supervisor applies the proposals via run.set_criterion()
        (e.g. cause "criteria engine after answer to sq_1.1.2") and
        writes them to the audit log.
        """
        current_states = run.criterion_states
        proposed: dict[str, CriterionStatus] = {}
        for criterion in self._definition.all_criteria():
            current = current_states[criterion.id]
            if current in (
                CriterionStatus.SATISFIED,
                CriterionStatus.NOT_SATISFIED,
            ):
                continue  # decided guard: no re-evaluation
            new_status = evaluate_criterion(
                criterion, readings, self._declarations
            )
            if new_status is not current:
                proposed[criterion.id] = new_status
        return proposed