"""
Unit tests for backend/interview_run.py (RECONSTRUCTION_PLAN.md Phase 2,
step 2.2): the InterviewRun state machine — the explicit legal/illegal
transition tables, the invariants (single RESERVED question, terminal
run guard, non-empty causes), the four-valued criterion logic, the
transition log / RunState snapshot and the audit-event projection —
plus the per-EPISODE navigation-value readings and criterion scoping
(design addendum 2026-10-31, OFQ-2), the declarative jump force-coding
(OFQ-7) and the OUT-OF-SCOPE exit (OFQ-6) as core state-machine
behavior (the step 2.6 core test suite).

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
    ValueReading,
    new_run_id,
    parse_event,
)
from interview_definition import (
    AnyRule,
    CriterionDefinition,
    CriterionScope,
    EqRule,
    EpisodeConfiguration,
    InterviewDefinition,
    PastEpisodesConfig,
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
        # The table allows CLOSED/SKIPPED -> REQUIRED, but the ENTRY
        # POINT decides: require_question is only for UNSEEN questions
        # (the first visit); a CLOSED/SKIPPED question is re-asked via
        # reopen_question (the episode model — see TestEpisodeReopen).
        require_guard = {QuestionState.CLOSED, QuestionState.SKIPPED}
        for from_state in QuestionState:
            for method_name, to_state in self.METHODS.items():
                with self.subTest(fro=from_state.value, to=to_state.value):
                    run = make_run()
                    drive_question(run, "sq_1.1.1", from_state)
                    if (
                        method_name == "require_question"
                        and from_state in require_guard
                    ):
                        with self.assertRaises(IllegalTransitionError):
                            run.require_question("sq_1.1.1", "test")
                        self.assertEqual(
                            run.question_state("sq_1.1.1"), from_state
                        )
                    elif to_state in table[from_state]:
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


def make_episode_run(
    episodes: EpisodeConfiguration | None = None,
) -> InterviewRun:
    """A run over the test definition with the episodes block replaced."""
    definition = make_definition()
    if episodes is not None:
        definition = definition.model_copy(update={"episodes": episodes})
    return InterviewRun(definition, new_run_id())


def dynamic_episodes() -> EpisodeConfiguration:
    return EpisodeConfiguration(
        enabled=True, past=PastEpisodesConfig(declaration="dynamic")
    )


def make_scoped_definition() -> InterviewDefinition:
    """
    A definition carrying one RUN-scoped criterion (no scope — the
    single global state, the legacy behavior), one `current`-scoped
    criterion (decided in the ACTIVE episode context) and one `past`-
    scoped criterion (decidable only in a past-episode context) — the
    episode-scoped model (design addendum 2026-10-31, OFQ-2). The
    episodes block is enabled (dynamic) so a past context can be
    declared and entered at run time.
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
                                id="c_run",
                                rule=EqRule(value="v_yes", equals=True),
                            ),
                            CriterionDefinition(
                                id="c_cur",
                                rule=EqRule(value="v_yes", equals=True),
                                scope=CriterionScope(episode="current"),
                            ),
                            CriterionDefinition(
                                id="c_past",
                                rule=EqRule(value="v_yes", equals=True),
                                scope=CriterionScope(episode="past"),
                            ),
                        ],
                    )
                ],
            )
        ],
        episodes=EpisodeConfiguration(
            enabled=True, past=PastEpisodesConfig(declaration="dynamic")
        ),
    )


class TestEpisodeRegistry(unittest.TestCase):
    """
    The per-interview episode registry (design addendum 2026-10-26):
    seeding, dynamic declaration (deterministic ep_1, ep_2, ...), the
    patient's verbatim aliases, and the Diagnostiker's merge.
    """

    def test_now_episode_is_always_seeded(self):
        run = make_episode_run()  # episodes disabled
        self.assertIn("now", run.episodes)
        now = run.episodes["now"]
        self.assertEqual(now.kind, "now")
        self.assertEqual(now.window_days, 30)
        self.assertIsNone(now.merged_into)

    def test_predeclared_episodes_are_seeded(self):
        run = make_episode_run(
            EpisodeConfiguration(
                enabled=True,
                past=PastEpisodesConfig(
                    declaration="predeclared", episodes=["e_one", "e_two"]
                ),
            )
        )
        self.assertEqual(set(run.episodes), {"now", "e_one", "e_two"})
        self.assertEqual(run.episodes["e_one"].kind, "past")
        self.assertEqual(run.episodes["e_one"].label, "e_one")
        self.assertEqual(run.canonical_episode("e_two"), "e_two")

    def test_declare_episode_needs_enabled_dynamic_mode(self):
        for episodes in (
            None,
            dynamic_episodes(),
            EpisodeConfiguration(
                enabled=True,
                past=PastEpisodesConfig(
                    declaration="predeclared", episodes=["e_one"]
                ),
            ),
        ):
            with self.subTest(episodes=episodes):
                run = make_episode_run(episodes)
                if episodes is None or (
                    episodes is not None
                    and episodes.past.declaration != "dynamic"
                ):
                    with self.assertRaises(IllegalTransitionError):
                        run.declare_episode("other period", "clinician")
                else:
                    run.declare_episode("other period", "clinician")

    def test_declare_episode_assigns_deterministic_ids(self):
        run = make_episode_run(dynamic_episodes())
        rec1 = run.declare_episode(
            "another period", "clinician declares a past episode"
        )
        rec2 = run.declare_episode(
            "   ", "clinician declares a second past episode"
        )
        self.assertEqual(
            (rec1.subject, rec1.subject_id), (SubjectKind.EPISODE, "ep_1")
        )
        self.assertEqual(
            (rec2.subject, rec2.subject_id), (SubjectKind.EPISODE, "ep_2")
        )
        self.assertEqual(rec1.from_state, "none")
        self.assertEqual(rec1.to_state, "declared")
        self.assertEqual(run.episodes["ep_1"].kind, "past")
        self.assertEqual(run.episodes["ep_1"].label, "another period")
        self.assertIsNone(run.episodes["ep_2"].label)  # blank label -> None


    def test_record_episode_alias_accumulates_on_the_canonical_episode(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        run.record_episode_alias("ep_1", "march 2026", "the patient specifies")
        run.record_episode_alias("ep_1", "march 2026", "the patient repeats")
        run.record_episode_alias("ep_1", "the month before uni", "the patient")
        self.assertEqual(
            run.episodes["ep_1"].aliases,
            ("march 2026", "the month before uni"),
        )
        records = run.transitions
        self.assertEqual(records[-1].subject, SubjectKind.EPISODE)
        self.assertEqual(records[-1].to_state, "aliased")

    def test_merge_episodes_unifies_to_the_earlier_declaration(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("period one", "clinician")
        run.record_episode_alias("ep_1", "march 2026", "the patient")
        run.declare_episode("period two", "clinician")
        run.record_episode_alias("ep_2", "the month before uni", "the patient")
        rec = run.merge_episodes(
            "ep_2", "ep_1", "Diagnostiker: the same real episode"
        )
        self.assertEqual(rec.subject_id, "ep_2")
        self.assertEqual(rec.to_state, "merged_into:ep_1")
        self.assertEqual(run.canonical_episode("ep_2"), "ep_1")
        self.assertEqual(run.canonical_episode("ep_1"), "ep_1")
        self.assertEqual(run.episodes["ep_2"].merged_into, "ep_1")
        self.assertIsNone(run.episodes["ep_1"].merged_into)
        # the aliases live on the winner (order preserved, no duplicates)
        self.assertEqual(
            run.episodes["ep_1"].aliases,
            ("march 2026", "the month before uni"),
        )
        # new aliases are recorded on the canonical episode via either id
        run.record_episode_alias("ep_2", "early spring", "the patient")
        self.assertIn("early spring", run.episodes["ep_1"].aliases)

    def test_merge_of_the_same_episode_is_rejected(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("period one", "clinician")
        run.declare_episode("period two", "clinician")
        run.merge_episodes("ep_1", "ep_2", "Diagnostiker: identical")
        with self.assertRaises(IllegalTransitionError):
            run.merge_episodes("ep_1", "ep_2", "again")

    def test_unknown_episode_ids_fail_loudly(self):
        run = make_episode_run(dynamic_episodes())
        with self.assertRaises(ValueError):
            run.canonical_episode("ep_9")

    def test_episode_operations_respect_the_terminal_run(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        run.abort_run("participant left")
        with self.assertRaises(IllegalTransitionError):
            run.declare_episode("another", "clinician")
        with self.assertRaises(IllegalTransitionError):
            run.record_episode_alias("ep_1", "march", "patient")

    def test_episode_operations_carry_causes(self):
        run = make_episode_run(dynamic_episodes())
        with self.assertRaises(ValueError):
            run.declare_episode("another period", "")


class TestEpisodeReopen(unittest.TestCase):
    """
    reopen_question: CLOSED/SKIPPED -> REQUIRED FOR ANOTHER EPISODE,
    with the (question, episode) once-only invariant (the plain
    "never re-asked" rule is the special case of it).
    """

    def test_require_records_the_now_episode(self):
        run = make_episode_run(dynamic_episodes())
        run.require_question("sq_1.1.1", "nav plan")
        self.assertEqual(run.question_episodes["sq_1.1.1"], ["now"])
        self.assertEqual(run.transitions[0].episode, "now")

    def test_reopen_closed_question_for_a_new_episode(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.CLOSED)
        run.declare_episode("another period", "clinician")
        rec = run.reopen_question(
            "sq_1.1.1", "ep_1", "episode jump re-asks the target group"
        )
        self.assertEqual(rec.from_state, "CLOSED")
        self.assertEqual(rec.to_state, "REQUIRED")
        self.assertEqual(rec.episode, "ep_1")
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.REQUIRED)
        self.assertEqual(run.question_episodes["sq_1.1.1"], ["now", "ep_1"])
        # the audit event carries the episode
        event = rec.to_event(run.run_id, DEFINITION_VERSION)
        self.assertEqual(event.episode, "ep_1")

    def test_reopen_from_skipped(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.SKIPPED)
        run.declare_episode("another period", "clinician")
        run.reopen_question("sq_1.1.1", "ep_1", "episode jump")
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.REQUIRED)

    def test_same_episode_is_never_re_asked(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.CLOSED)
        with self.assertRaises(IllegalTransitionError) as ctx:
            run.reopen_question("sq_1.1.1", "now", "episode jump")
        self.assertIn("never re-asked for the same episode", str(ctx.exception))
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.CLOSED)

    def test_same_real_episode_after_a_merge_is_never_re_asked(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.CLOSED)
        run.declare_episode("period one", "clinician")
        run.record_episode_alias("ep_1", "march 2026", "the patient")
        run.declare_episode("period two", "clinician")
        run.merge_episodes("ep_2", "ep_1", "Diagnostiker: identical")
        run.reopen_question("sq_1.1.1", "ep_1", "episode jump")
        # work the re-opened question through to CLOSED again
        run.reserve_question("sq_1.1.1", "nav")
        run.deliver_question("sq_1.1.1", "output")
        run.record_answer("sq_1.1.1", "answered for ep_1")
        run.close_question("sq_1.1.1", "closed for ep_1")
        # ep_2 is the same REAL episode (merged into ep_1) -> rejected
        with self.assertRaises(IllegalTransitionError) as ctx:
            run.reopen_question("sq_1.1.1", "ep_2", "episode jump again")
        self.assertIn("already asked for episode 'ep_1'", str(ctx.exception))
        # the rejected re-open leaves the question where it was
        self.assertEqual(run.question_state("sq_1.1.1"), QuestionState.CLOSED)

    def test_reopen_only_from_terminal_question_states(self):
        run = make_episode_run(dynamic_episodes())
        with self.assertRaises(IllegalTransitionError):
            run.reopen_question("sq_1.1.1", "ep_1", "x")  # UNSEEN
        run.require_question("sq_1.1.1", "nav plan")
        with self.assertRaises(IllegalTransitionError):
            run.reopen_question("sq_1.1.1", "ep_1", "x")  # REQUIRED

    def test_require_from_closed_is_rejected(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.CLOSED)
        with self.assertRaises(IllegalTransitionError) as ctx:
            run.require_question("sq_1.1.1", "nav plan again")
        self.assertIn("reopen_question", str(ctx.exception))

    def test_snapshot_carries_the_episode_data(self):
        run = make_episode_run(dynamic_episodes())
        drive_question(run, "sq_1.1.1", QuestionState.CLOSED)
        run.declare_episode("another period", "clinician")
        run.record_episode_alias("ep_1", "march 2026", "the patient")
        run.reopen_question("sq_1.1.1", "ep_1", "episode jump")
        snap = run.snapshot(run_version=9)
        self.assertEqual(snap.question_episodes["sq_1.1.1"], ["now", "ep_1"])
        self.assertEqual(set(snap.episodes), {"now", "ep_1"})
        self.assertEqual(snap.episodes["ep_1"].aliases, ("march 2026",))
        # the EpisodeState value objects are FROZEN: the snapshot's
        # registry cannot be mutated in place
        with self.assertRaises(Exception):
            snap.episodes["ep_1"].aliases = ("tampered",)  # type: ignore[misc]


class TestValueReadingsAndEpisodeContext(unittest.TestCase):
    """
    The per-EPISODE navigation-value readings (design addendum
    2026-10-31, OFQ-2): a reading is the CURRENT content of a value in
    ONE episode's context — a STORE operation, not a state transition
    (there is no value subject in the audit vocabulary). The criteria
    engine evaluates each criterion only against the readings its scope
    selects, so one episode's values cannot contaminate another's
    decisions.
    """

    def test_set_value_goes_into_the_active_context(self):
        run = make_episode_run(dynamic_episodes())
        reading = ValueReading(name="v_yes", value=True)
        run.set_value(reading, "Diagnostiker: answer to sq_1.1.1")
        self.assertEqual(run.readings()["v_yes"], reading)
        self.assertEqual(run.readings("now")["v_yes"], reading)
        self.assertEqual(run.value("v_yes"), reading)
        self.assertIsNone(run.value("v_missing"))
        # only the now context carries readings so far
        self.assertEqual(set(run.readings_by_context()), {"now"})

    def test_set_value_follows_the_active_context(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        run.set_active_episode("ep_1", "re-enter the interview for ep_1")
        run.set_value(
            ValueReading(name="v_yes", value=True), "answered ep_1"
        )
        # the reading landed in ep_1 — the ACTIVE context ...
        self.assertTrue(run.readings()["v_yes"].value)
        # ... and the now-episode's context is untouched
        self.assertEqual(run.readings("now"), {})
        self.assertIn("v_yes", run.readings("ep_1"))

    def test_set_value_for_targets_the_named_context(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        run.set_value_for(
            "ep_1", ValueReading(name="v_yes", value=True), "answered ep_1"
        )
        self.assertTrue(run.readings("ep_1")["v_yes"].value)
        self.assertEqual(run.readings("now"), {})
        # the active context is still "now" — set_value_for does not
        # switch contexts
        self.assertEqual(run.active_episode, "now")
        self.assertEqual(run.readings(), {})

    def test_latest_reading_wins(self):
        run = make_episode_run(dynamic_episodes())
        run.set_value(ValueReading(name="v_yes", value=True), "first answer")
        run.set_value(
            ValueReading(name="v_yes", value=False), "clarification"
        )
        self.assertFalse(run.readings()["v_yes"].value)

    def test_readings_are_copies(self):
        run = make_episode_run(dynamic_episodes())
        run.set_value(ValueReading(name="v_yes", value=True), "x")
        one_context = run.readings()
        one_context["v_tampered"] = ValueReading(
            name="v_tampered", value=1
        )
        del one_context["v_yes"]
        by_context = run.readings_by_context()
        by_context["now"]["v_tampered"] = ValueReading(
            name="v_tampered", value=1
        )
        # the run's own state survived both tampering attempts
        self.assertTrue(run.readings()["v_yes"].value)
        self.assertNotIn("v_tampered", run.readings())
        self.assertNotIn("v_tampered", run.readings("now"))

    def test_all_readings_merge_contexts_in_episode_order(self):
        run = make_episode_run(dynamic_episodes())
        # the now-episode answers yes, the past episode re-answers no
        run.set_value_for(
            "now", ValueReading(name="v_yes", value=True), "now"
        )
        run.declare_episode("another period", "clinician")
        run.set_value_for(
            "ep_1", ValueReading(name="v_yes", value=False), "ep_1"
        )
        merged = run.all_readings()
        # the LATER context (ep_1, later in the registry order)
        # overrides the earlier one — the legacy single-mapping view
        self.assertFalse(merged["v_yes"].value)
        # each context keeps its own reading (the isolation the
        # criteria engine relies on)
        self.assertTrue(run.readings("now")["v_yes"].value)
        self.assertFalse(run.readings("ep_1")["v_yes"].value)
        self.assertEqual(set(merged), {"v_yes"})

    def test_set_value_guards(self):
        run = make_episode_run(dynamic_episodes())
        with self.assertRaises(ValueError):  # non-empty cause
            run.set_value(ValueReading(name="v_yes", value=True), "   ")
        with self.assertRaises(ValueError):  # unknown episode id
            run.set_value_for(
                "ep_9", ValueReading(name="v_yes", value=True), "x"
            )
        run.abort_run("participant left")
        with self.assertRaises(IllegalTransitionError):  # terminal run
            run.set_value(ValueReading(name="v_yes", value=True), "late")
        with self.assertRaises(IllegalTransitionError):
            run.set_value_for(
                "now", ValueReading(name="v_yes", value=True), "late"
            )

    def test_set_active_episode_records_a_logged_transition(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        record = run.set_active_episode("ep_1", "re-enter for the past")
        self.assertEqual(record.subject, SubjectKind.EPISODE)
        self.assertEqual(record.subject_id, "ep_1")
        self.assertEqual(
            (record.from_state, record.to_state),
            ("active:now", "active:ep_1"),
        )
        self.assertEqual(record.cause, "re-enter for the past")
        self.assertEqual(run.active_episode, "ep_1")
        # switching back is legal and logged as well
        record2 = run.set_active_episode("now", "back to the now-episode")
        self.assertEqual(
            (record2.from_state, record2.to_state),
            ("active:ep_1", "active:now"),
        )
        self.assertEqual(run.active_episode, "now")

    def test_set_active_episode_guards(self):
        run = make_episode_run(dynamic_episodes())
        with self.assertRaises(ValueError):  # not in the registry
            run.set_active_episode("ep_9", "x")
        with self.assertRaises(ValueError):  # non-empty cause
            run.set_active_episode("now", "   ")
        run.abort_run("participant left")
        with self.assertRaises(IllegalTransitionError):  # terminal run
            run.set_active_episode("now", "late")


class TestEpisodeScopedCriteria(unittest.TestCase):
    """
    Episode-scoped criteria (design addendum 2026-10-31, OFQ-2):
    a `current`-scoped criterion is decided IN the active episode
    context (the SAME transition table per context — a criterion
    undecided in one context is OPEN there, whatever the other
    contexts say); a `past`-scoped criterion can never be decided in
    the now-episode context (its values are not asked there). RUN-
    scoped criteria keep the single global state and are NOT part of
    the per-context maps.
    """

    def make_run(self) -> InterviewRun:
        return InterviewRun(make_scoped_definition(), new_run_id())

    def declare_and_enter_ep_1(self, run: InterviewRun) -> None:
        run.declare_episode("another period", "clinician")
        run.set_active_episode("ep_1", "re-enter for ep_1")

    def test_current_scoped_criterion_is_decided_per_context(self):
        run = self.make_run()
        self.assertEqual(run.criterion_state("c_cur"), CriterionStatus.OPEN)
        run.set_criterion("c_cur", CriterionStatus.SATISFIED, "decided now")
        self.assertEqual(
            run.criterion_state("c_cur"), CriterionStatus.SATISFIED
        )
        self.assertEqual(
            run.episode_criterion_states("now"),
            {"c_cur": CriterionStatus.SATISFIED},
        )
        # entering a past context: undecided there -> OPEN there
        self.declare_and_enter_ep_1(run)
        self.assertEqual(run.criterion_state("c_cur"), CriterionStatus.OPEN)
        self.assertEqual(run.episode_criterion_states(), {})  # active: ep_1
        # ... while the now-episode keeps its decision
        self.assertEqual(
            run.episode_criterion_states("now"),
            {"c_cur": CriterionStatus.SATISFIED},
        )

    def test_transition_table_holds_per_context(self):
        run = self.make_run()
        self.declare_and_enter_ep_1(run)
        run.set_criterion("c_cur", CriterionStatus.NOT_SATISFIED, "ep_1")
        # decided -> decided is illegal IN THIS CONTEXT, too ...
        with self.assertRaises(IllegalTransitionError):
            run.set_criterion("c_cur", CriterionStatus.SATISFIED, "direct")
        # ... but legal via UNCLEAR (the only move out of a decided state)
        run.set_criterion("c_cur", CriterionStatus.UNCLEAR, "misread")
        run.set_criterion("c_cur", CriterionStatus.SATISFIED, "re-decided")
        self.assertEqual(
            run.criterion_state("c_cur"), CriterionStatus.SATISFIED
        )
        self.assertEqual(
            run.episode_criterion_states("ep_1"),
            {"c_cur": CriterionStatus.SATISFIED},
        )

    def test_run_scoped_criterion_ignores_the_context(self):
        run = self.make_run()
        run.set_criterion("c_run", CriterionStatus.SATISFIED, "decided")
        self.declare_and_enter_ep_1(run)
        # run-scoped: the single global state, whatever the context
        self.assertEqual(
            run.criterion_state("c_run"), CriterionStatus.SATISFIED
        )
        # ... and it is NOT part of any per-context map
        self.assertNotIn("c_run", run.episode_criterion_states("now"))
        self.assertNotIn("c_run", run.episode_criterion_states())

    def test_past_scoped_criterion_cannot_be_decided_in_the_now_episode(self):
        run = self.make_run()
        with self.assertRaises(IllegalTransitionError) as ctx:
            run.set_criterion("c_past", CriterionStatus.SATISFIED, "early")
        self.assertIn("past-episode context", str(ctx.exception))
        self.assertEqual(run.criterion_state("c_past"), CriterionStatus.OPEN)

    def test_past_scoped_criterion_is_decided_in_a_past_context(self):
        run = self.make_run()
        self.declare_and_enter_ep_1(run)
        run.set_criterion("c_past", CriterionStatus.SATISFIED, "decided")
        self.assertEqual(
            run.criterion_state("c_past"), CriterionStatus.SATISFIED
        )
        self.assertEqual(
            run.episode_criterion_states("ep_1"),
            {"c_past": CriterionStatus.SATISFIED},
        )
        # back in the now-episode it is undecided again — and cannot be
        # decided there at all (the past-scoped guard)
        run.set_active_episode("now", "back to the now-episode")
        self.assertEqual(run.criterion_state("c_past"), CriterionStatus.OPEN)
        with self.assertRaises(IllegalTransitionError):
            run.set_criterion("c_past", CriterionStatus.NOT_SATISFIED, "now")

    def test_episode_criterion_states_returns_a_copy(self):
        run = self.make_run()
        run.set_criterion("c_cur", CriterionStatus.SATISFIED, "x")
        states = run.episode_criterion_states("now")
        states["c_tampered"] = CriterionStatus.SATISFIED
        self.assertEqual(
            run.episode_criterion_states("now"),
            {"c_cur": CriterionStatus.SATISFIED},
        )

    def test_scoped_decision_carry_the_context_in_the_log(self):
        run = self.make_run()
        run.set_criterion("c_cur", CriterionStatus.SATISFIED, "decided now")
        record = run.transitions[-1]
        self.assertEqual(record.subject, SubjectKind.CRITERION)
        self.assertEqual(record.subject_id, "c_cur")
        self.assertEqual(record.episode, "now")


class TestForceCriterion(unittest.TestCase):
    """
    run.force_criterion (design addendum 2026-11-03, OFQ-7): the
    declarative jump force-coding ("code A30 with '+' and A38 with
    'JA'") is applied by the Supervisor as a CODING decision in the
    ACTIVE episode context — not a derived one. The transition table
    forbids a direct decided -> decided move, so an already-decided
    criterion is reopened via UNCLEAR first (both steps are logged);
    a criterion that already holds the target status is a no-op (no
    records, no log entry).
    """

    def test_open_criterion_is_forced_with_a_single_record(self):
        run = make_run()
        records = run.force_criterion(
            "c_one", CriterionStatus.SATISFIED, "A52 escalation: code +'"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].subject, SubjectKind.CRITERION)
        self.assertEqual(records[0].subject_id, "c_one")
        self.assertEqual(
            (records[0].from_state, records[0].to_state),
            ("OPEN", "SATISFIED"),
        )
        self.assertEqual(records[0].cause, "A52 escalation: code +'")
        self.assertEqual(run.criterion_state("c_one"), CriterionStatus.SATISFIED)

    def test_unclear_criterion_is_forced_with_a_single_record(self):
        run = make_run()
        run.set_criterion("c_one", CriterionStatus.UNCLEAR, "misread")
        records = run.force_criterion(
            "c_one", CriterionStatus.NOT_SATISFIED, "re-coded"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            (records[0].from_state, records[0].to_state),
            ("UNCLEAR", "NOT_SATISFIED"),
        )
        self.assertEqual(
            run.criterion_state("c_one"), CriterionStatus.NOT_SATISFIED
        )

    def test_already_at_the_target_status_is_a_no_op(self):
        run = make_run()
        run.set_criterion("c_one", CriterionStatus.SATISFIED, "decided")
        before = len(run.transitions)
        records = run.force_criterion(
            "c_one", CriterionStatus.SATISFIED, "escalation fires again"
        )
        self.assertEqual(records, [])
        self.assertEqual(len(run.transitions), before)  # nothing logged
        self.assertEqual(
            run.criterion_state("c_one"), CriterionStatus.SATISFIED
        )

    def test_decided_criterion_is_reopened_via_unclear_first(self):
        run = make_run()
        run.set_criterion("c_one", CriterionStatus.SATISFIED, "decided")
        records = run.force_criterion(
            "c_one", CriterionStatus.NOT_SATISFIED, "A52: reclassify"
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(
            (records[0].from_state, records[0].to_state),
            ("SATISFIED", "UNCLEAR"),
        )
        self.assertEqual(records[0].cause, "A52: reclassify (reopened)")
        self.assertEqual(
            (records[1].from_state, records[1].to_state),
            ("UNCLEAR", "NOT_SATISFIED"),
        )
        self.assertEqual(records[1].cause, "A52: reclassify")
        self.assertEqual(
            run.criterion_state("c_one"), CriterionStatus.NOT_SATISFIED
        )

    def test_force_coding_happens_in_the_active_context(self):
        run = InterviewRun(make_scoped_definition(), new_run_id())
        run.set_criterion("c_cur", CriterionStatus.SATISFIED, "decided now")
        run.declare_episode("another period", "clinician")
        run.set_active_episode("ep_1", "re-enter for ep_1")
        # c_cur is OPEN in ep_1: the force-coding lands THERE
        records = run.force_criterion(
            "c_cur", CriterionStatus.SATISFIED, "escalation in ep_1"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            run.episode_criterion_states("ep_1"),
            {"c_cur": CriterionStatus.SATISFIED},
        )
        # and the now-episode keeps its own decision
        run.set_active_episode("now", "back to the now-episode")
        self.assertEqual(
            run.episode_criterion_states("now"),
            {"c_cur": CriterionStatus.SATISFIED},
        )

    def test_force_criterion_guards(self):
        run = make_run()
        with self.assertRaises(ValueError):  # unknown criterion id
            run.force_criterion("c_zzz", CriterionStatus.SATISFIED, "x")
        with self.assertRaises(ValueError):  # non-empty cause
            run.force_criterion("c_one", CriterionStatus.SATISFIED, "   ")
        run.complete_run("nothing more to ask")
        with self.assertRaises(IllegalTransitionError):  # terminal run
            run.force_criterion("c_one", CriterionStatus.SATISFIED, "late")


class TestOutOfScopeExit(unittest.TestCase):
    """
    The OUT-OF-SCOPE exit (design addendum 2026-10-31, OFQ-6): the
    interview's routed continuation target lies OUTSIDE the loaded
    definition (the SCID excerpt ends at A78). The run is ABORTED with
    the OutOfScopeExit recorded on it — an explicit, logged hand-over,
    never a silently swallowed jump. The snapshot surfaces both the
    exit and the active episode context.
    """

    def test_abort_out_of_scope_aborts_and_records_the_exit(self):
        run = make_run()
        record = run.abort_out_of_scope(
            "sec_a78", note="  SCID continues beyond this excerpt  "
        )
        self.assertIs(run.status, RunStatus.ABORTED)
        self.assertEqual(
            (record.subject, record.subject_id), (SubjectKind.RUN, "")
        )
        self.assertEqual(
            (record.from_state, record.to_state), ("ACTIVE", "ABORTED")
        )
        # the default cause names the external target
        self.assertIn("out-of-scope exit", record.cause)
        self.assertIn("sec_a78", record.cause)
        self.assertIn("(outside this definition)", record.cause)
        exit_ = run.out_of_scope
        self.assertIsNotNone(exit_)
        self.assertEqual(exit_.target_ref, "sec_a78")
        self.assertEqual(exit_.note, "SCID continues beyond this excerpt")

    def test_custom_cause_replaces_the_default(self):
        run = make_run()
        record = run.abort_out_of_scope("sec_a78", cause="routed outside")
        self.assertEqual(record.cause, "routed outside")

    def test_blank_note_is_stored_as_none(self):
        run = make_run()
        run.abort_out_of_scope("sec_a78", note="   ")
        self.assertEqual(run.out_of_scope.target_ref, "sec_a78")
        self.assertIsNone(run.out_of_scope.note)

    def test_out_of_scope_requires_an_active_run(self):
        run = make_run()
        run.complete_run("nothing more to ask")
        with self.assertRaises(IllegalTransitionError):
            run.abort_out_of_scope("sec_a78")
        aborted = make_run()
        aborted.abort_run("participant left")
        with self.assertRaises(IllegalTransitionError):
            aborted.abort_out_of_scope("sec_a78")

    def test_run_is_terminal_after_the_exit(self):
        run = make_run()
        run.abort_out_of_scope("sec_a78")
        with self.assertRaises(IllegalTransitionError):
            run.require_question("sq_1.1.1", "too late")
        with self.assertRaises(IllegalTransitionError):
            run.set_criterion("c_one", CriterionStatus.SATISFIED, "late")
        with self.assertRaises(IllegalTransitionError):
            run.abort_out_of_scope("sec_a78")  # a second exit is impossible
        with self.assertRaises(IllegalTransitionError):
            run.complete_run("too late")

    def test_snapshot_carries_the_exit_and_the_active_episode(self):
        run = make_run()
        self.assertIsNone(run.out_of_scope)
        self.assertIsNone(run.snapshot().out_of_scope)
        self.assertEqual(run.snapshot().active_episode, "now")
        run.abort_out_of_scope("sec_a78", note="hand-over to the full SCID")
        snap = run.snapshot(run_version=7)
        self.assertIs(snap.status, RunStatus.ABORTED)
        self.assertEqual(snap.out_of_scope.target_ref, "sec_a78")
        self.assertEqual(snap.out_of_scope.note, "hand-over to the full SCID")
        self.assertEqual(snap.active_episode, "now")
        self.assertEqual(snap.run_version, 7)

    def test_snapshot_carries_the_active_episode_after_a_switch(self):
        run = make_episode_run(dynamic_episodes())
        run.declare_episode("another period", "clinician")
        run.set_active_episode("ep_1", "re-enter for ep_1")
        self.assertEqual(run.snapshot().active_episode, "ep_1")
        run.set_active_episode("now", "back to the now-episode")
        self.assertEqual(run.snapshot().active_episode, "now")
