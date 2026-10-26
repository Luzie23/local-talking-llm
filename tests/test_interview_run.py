"""
Unit tests for backend/interview_run.py (RECONSTRUCTION_PLAN.md Phase 2,
step 2.2): the InterviewRun state machine — the explicit legal/illegal
transition tables, the invariants (single RESERVED question, terminal
run guard, non-empty causes), the four-valued criterion logic, the
transition log / RunState snapshot and the audit-event projection.

All tests are deterministic and need no network access.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from contracts import (
    CriterionStatus,
    QuestionState,
    RunStatus,
    StateTransitionEvent,
    SubjectKind,
    new_run_id,
    parse_event,
)
from interview_definition import (
    AnyRule,
    CriterionDefinition,
    EqRule,
    InterviewDefinition,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ValueDeclaration,
)
from interview_run import IllegalTransitionError, InterviewRun

DEFINITION_VERSION = "test_guide@v1"


def make_definition() -> InterviewDefinition:
    """
    One v2 definition with two questions and two criteria. The criteria
    live on the (criteria) question group, per the hierarchical format.
    """
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="test_guide",
        title="Test Guide",
        values=[ValueDeclaration(name="v_yes", type="bool")],
        sections=[
            SectionGroup(
                id="sec_1",
                title="A",
                children=[
                    QuestionGroup(
                        id="qg_1.1",
                        title="A",
                        strategy="all_required",
                        criteria_group=True,
                        questions=[
                            QuestionDefinition(id="sq_1.1.1", text="A?"),
                            QuestionDefinition(id="sq_1.1.2", text="B?"),
                        ],
                        criteria=[
                            CriterionDefinition(
                                id="c_one", rule=EqRule(value="v_yes", equals=True)
                            ),
                            CriterionDefinition(
                                id="c_two",
                                rule=AnyRule(
                                    rules=[EqRule(value="v_yes", equals=False)]
                                ),
                            ),
                        ],
                    )
                ],
            )
        ],
    )


def make_run() -> InterviewRun:
    return InterviewRun(make_definition(), new_run_id())


def drive_question(run: InterviewRun, question_id: str, target: QuestionState) -> None:
    """Drive one question along the happy path to `target` (helper for the matrix)."""
    if target is QuestionState.UNSEEN:
        return
    if target is QuestionState.SKIPPED:
        run.require_question(question_id, "drive")
        run.skip_question(question_id, "drive")
        return
    run.require_question(question_id, "drive")
    if target is QuestionState.REQUIRED:
        return
    run.reserve_question(question_id, "drive")
    if target is QuestionState.RESERVED:
        return
    run.deliver_question(question_id, "drive")
    if target is QuestionState.DELIVERED:
        return
    run.record_answer(question_id, "drive")
    if target is QuestionState.ANSWER_RECORDED:
        return
    if target is QuestionState.CLOSED:
        run.close_question(question_id, "drive")
        return
    run.revise_question(question_id, "drive")
    if target is QuestionState.REVISED:
        return
    raise AssertionError(f"unhandled target {target}")




def drive_criterion(
    run: InterviewRun, criterion_id: str, target: CriterionStatus
) -> None:
    """OPEN is the initial state; the other three are all legal from OPEN."""
    if target is CriterionStatus.OPEN:
        return
    run.set_criterion(criterion_id, target, "drive")


class TestQuestionTransitionMatrix(unittest.TestCase):
    """Every (from_state, method) pair of the question table is tested exactly once."""

    METHODS = {
        "require_question": QuestionState.REQUIRED,
        "reserve_question": QuestionState.RESERVED,
        "deliver_question": QuestionState.DELIVERED,
        "record_answer": QuestionState.ANSWER_RECORDED,
        "close_question": QuestionState.CLOSED,
        "revise_question": QuestionState.REVISED,
        "skip_question": QuestionState.SKIPPED,
    }

    def test_full_matrix(self):
        import interview_run

        table = interview_run._QUESTION_TRANSITIONS
        for from_state in QuestionState:
            for method_name, to_state in self.METHODS.items():
                with self.subTest(fro=from_state.value, to=to_state.value):
                    run = make_run()
                    drive_question(run, "sq_1.1.1", from_state)
                    if to_state in table[from_state]:
                        record = getattr(run, method_name)("sq_1.1.1", "test")
                        self.assertEqual(run.question_state("sq_1.1.1"), to_state)
                        self.assertEqual(record.from_state, from_state.value)
                        self.assertEqual(record.to_state, to_state.value)
                    else:
                        with self.assertRaises(IllegalTransitionError):
                            getattr(run, method_name)("sq_1.1.1", "test")
                        # state is unchanged after a rejected transition
                        self.assertEqual(run.question_state("sq_1.1.1"), from_state)


class TestCriterionTransitionMatrix(unittest.TestCase):
    """The four-valued logic: decided -> UNCLEAR only, never decided -> decided."""

    def test_full_matrix(self):
        import interview_run

        table = interview_run._CRITERION_TRANSITIONS
        for from_state in CriterionStatus:
            for to_state in CriterionStatus:
                with self.subTest(fro=from_state.value, to=to_state.value):
                    run = make_run()
                    drive_criterion(run, "c_one", from_state)
                    if to_state in table[from_state]:
                        record = run.set_criterion("c_one", to_state, "test")
                        self.assertEqual(run.criterion_state("c_one"), to_state)
                        self.assertEqual(record.from_state, from_state.value)
                        self.assertEqual(record.to_state, to_state.value)
                    else:
                        with self.assertRaises(IllegalTransitionError):
                            run.set_criterion("c_one", to_state, "test")
                        self.assertEqual(run.criterion_state("c_one"), from_state)

    def test_decided_criteria_view(self):
        run = make_run()
        self.assertEqual(run.decided_criteria(), {})
        run.set_criterion("c_one", CriterionStatus.SATISFIED, "diag")
        run.set_criterion("c_two", CriterionStatus.NOT_SATISFIED, "diag")
        run.set_criterion("c_one", CriterionStatus.UNCLEAR, "re-opened")
        # c_two stays decided, c_one is back to UNCLEAR
        self.assertEqual(
            run.decided_criteria(), {"c_two": CriterionStatus.NOT_SATISFIED}
        )


class TestRunInvariants(unittest.TestCase):
    """The invariants enforced beyond the transition tables."""

    def test_at_most_one_reserved_question(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav")
        run.require_question("sq_1.1.2", "nav")
        run.reserve_question("sq_1.1.1", "nav")
        self.assertEqual(run.reserved_question(), "sq_1.1.1")
        with self.assertRaises(IllegalTransitionError) as ctx:
            run.reserve_question("sq_1.1.2", "nav")
        self.assertIn("RESERVED", str(ctx.exception))
        # closing the reserved question frees the slot
        run.deliver_question("sq_1.1.1", "nav")
        run.record_answer("sq_1.1.1", "nav")
        run.close_question("sq_1.1.1", "nav")
        run.reserve_question("sq_1.1.2", "nav")
        self.assertEqual(run.reserved_question(), "sq_1.1.2")

    def test_terminal_run_guard(self):
        run = make_run()
        run.complete_run("all questions closed")
        self.assertIs(run.status, RunStatus.COMPLETED)
        for attempt in (
            lambda: run.require_question("sq_1.1.1", "late"),
            lambda: run.set_criterion("c_one", CriterionStatus.SATISFIED, "late"),
            lambda: run.abort_run("late"),
            lambda: run.complete_run("again"),
        ):
            with self.assertRaises(IllegalTransitionError):
                attempt()
        # the question/criterion states are frozen in place
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.UNSEEN)
        self.assertEqual(run.criterion_state("c_one"), CriterionStatus.OPEN)

    def test_abort_also_terminal(self):
        run = make_run()
        run.abort_run("participant left")
        self.assertIs(run.status, RunStatus.ABORTED)
        with self.assertRaises(IllegalTransitionError):
            run.complete_run("too late")

    def test_run_id_scheme_enforced(self):
        with self.assertRaises(ValueError):
            InterviewRun(make_definition(), "bogus")

    def test_causes_must_be_non_empty(self):
        run = make_run()
        for cause in ("", "   "):
            with self.assertRaises(ValueError):
                run.require_question("sq_1.1.1", cause)
            with self.assertRaises(ValueError):
                run.set_criterion("c_one", CriterionStatus.SATISFIED, cause)
            with self.assertRaises(ValueError):
                run.complete_run(cause)
        self.assertEqual(len(run.transitions), 0)  # nothing was logged

    def test_unknown_ids_rejected(self):
        run = make_run()
        with self.assertRaises(ValueError) as ctx:
            run.require_question("q_zzz", "nav")
        self.assertIn("q_zzz", str(ctx.exception))
        with self.assertRaises(ValueError):
            run.set_criterion("c_zzz", CriterionStatus.SATISFIED, "diag")
        with self.assertRaises(ValueError):
            run.question_state("q_zzz")
        with self.assertRaises(ValueError):
            run.criterion_state("c_zzz")


class TestClarificationCycle(unittest.TestCase):
    """REVISED is a first-class state: answer missing -> re-ask -> record -> close."""

    def test_full_cycle(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav")
        run.reserve_question("sq_1.1.1", "nav")
        run.deliver_question("sq_1.1.1", "nav")
        run.record_answer("sq_1.1.1", "participant")
        self.assertIsNone(run.awaiting_answer())
        run.revise_question("sq_1.1.1", "answer lacks information")
        self.assertTrue(run.has_revised_question())
        self.assertEqual(run.revised_questions(), ["sq_1.1.1"])
        # the re-ask: REVISED -> DELIVERED (and only there)
        run.deliver_question("sq_1.1.1", "nav")
        self.assertEqual(run.awaiting_answer(), "sq_1.1.1")
        run.record_answer("sq_1.1.1", "participant")
        run.close_question("sq_1.1.1", "complete")
        self.assertFalse(run.has_revised_question())
        states = [r.to_state for r in run.transitions]
        self.assertEqual(
            states,
            ["REQUIRED", "RESERVED", "DELIVERED", "ANSWER_RECORDED",
             "REVISED", "DELIVERED", "ANSWER_RECORDED", "CLOSED"],
        )


class TestSkipSemantics(unittest.TestCase):
    """
    SKIPPED: the question is never delivered (the participant already
    gave us the information), so the criteria depending on it stay
    evaluable — but the interview itself continues: a skip is a
    question-level transition, never a run-level one.
    """

    def test_skip_does_not_end_the_interview(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav")
        run.skip_question("sq_1.1.1", "patient already stated this earlier")
        self.assertIs(run.question_state("sq_1.1.1"), QuestionState.SKIPPED)
        # the run stays ACTIVE: skipping is a question-level transition,
        # never a run-level one
        self.assertIs(run.status, RunStatus.ACTIVE)
        # a skipped question is locked for the run (it is never delivered)
        with self.assertRaises(IllegalTransitionError):
            run.deliver_question("sq_1.1.1", "too late")
        # the interview continues with the remaining questions
        run.require_question("sq_1.1.2", "nav")
        run.reserve_question("sq_1.1.2", "nav")
        run.deliver_question("sq_1.1.2", "output layer")
        self.assertIs(run.status, RunStatus.ACTIVE)
        # and the run still ends the normal way — explicitly, by decision
        run.record_answer("sq_1.1.2", "participant answered")
        run.close_question("sq_1.1.2", "no information missing")
        run.complete_run("nothing more to ask")
        self.assertIs(run.status, RunStatus.COMPLETED)


class TestTransitionLogAndSnapshot(unittest.TestCase):
    """The transition log, the RunState snapshot and the audit events."""

    def test_happy_path_log_and_snapshot(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav plan")
        run.reserve_question("sq_1.1.1", "nav plan")
        run.deliver_question("sq_1.1.1", "output layer")
        run.record_answer("sq_1.1.1", "participant answered")
        run.close_question("sq_1.1.1", "no information missing")
        records = run.transitions
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0].subject, SubjectKind.QUESTION)
        self.assertEqual(records[0].subject_id, "sq_1.1.1")
        self.assertEqual(records[0].from_state, "UNSEEN")
        self.assertEqual(records[0].to_state, "REQUIRED")
        self.assertEqual(records[0].cause, "nav plan")
        self.assertIsNotNone(records[0].timestamp.tzinfo)  # aware UTC
        # the snapshot is the value projection the Supervisor persists
        snap = run.snapshot(run_version=3)
        self.assertEqual(snap.run_id, run.run_id)
        self.assertIs(snap.status, RunStatus.ACTIVE)
        self.assertEqual(snap.definition_version, DEFINITION_VERSION)
        self.assertEqual(snap.run_version, 3)
        self.assertEqual(snap.question_states["sq_1.1.1"], QuestionState.CLOSED)
        self.assertEqual(snap.question_states["sq_1.1.2"], QuestionState.UNSEEN)
        self.assertEqual(snap.criterion_states["c_one"], CriterionStatus.OPEN)
        # snapshots are frozen value objects
        with self.assertRaises(Exception):
            snap.run_version = 4  # type: ignore[misc]
        # the snapshot status tracks the run
        run.complete_run("all closed")
        self.assertIs(run.snapshot().status, RunStatus.COMPLETED)

    def test_read_only_copies(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav")
        qs = run.question_states
        qs["sq_1.1.1"] = QuestionState.CLOSED  # mutate the copy
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.REQUIRED)
        cs = run.criterion_states
        cs["c_one"] = CriterionStatus.SATISFIED
        self.assertEqual(run.criterion_state("c_one"), CriterionStatus.OPEN)
        # the log is handed out as an immutable copy (a new tuple each time)
        logs = run.transitions
        self.assertIsInstance(logs, tuple)
        run.reserve_question("sq_1.1.1", "nav")
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(run.transitions), 2)

    def test_event_projection_and_roundtrip(self):
        run = make_run()
        run.require_question("sq_1.1.1", "nav plan")
        record = run.transitions[0]
        event = record.to_event(run.run_id, DEFINITION_VERSION)
        self.assertIsInstance(event, StateTransitionEvent)
        self.assertEqual(event.run_id, run.run_id)
        self.assertEqual(event.definition_version, DEFINITION_VERSION)
        self.assertEqual(event.event_type, "state_transition")
        self.assertIs(event.subject, SubjectKind.QUESTION)
        self.assertEqual(event.subject_id, "sq_1.1.1")
        self.assertEqual((event.from_state, event.to_state), ("UNSEEN", "REQUIRED"))
        self.assertEqual(event.cause, "nav plan")
        self.assertIsNone(event.event_id)  # assigned by the EventStore
        # the event round-trips through the event-log parser
        parsed = parse_event(event.model_dump())
        self.assertEqual(parsed, event)

    def test_run_level_event_has_no_subject_id(self):
        run = make_run()
        run.abort_run("participant left")
        event = run.transitions[0].to_event(run.run_id, DEFINITION_VERSION)
        self.assertIs(event.subject, SubjectKind.RUN)
        self.assertIsNone(event.subject_id)
        self.assertEqual((event.from_state, event.to_state), ("ACTIVE", "ABORTED"))
