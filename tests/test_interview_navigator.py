"""
Unit tests for backend/interview_navigator.py (RECONSTRUCTION_PLAN.md
Phase 2, step 2.4): the deterministic interview navigator — the
deterministic postorder fallback order, the outstanding/open question
sets, the candidate set handed to the interviewer agent (contracts.
QuestionCandidateSet with per-candidate reasons, including the
episodes a question was asked about), DEFERRED jump resolution at
GROUP COMPLETION (complete_group / group_is_complete; resolve_jump
mid-group fails loudly; the first declared fired jump wins; forward
jumps propose skips for bypassed open questions, backward jumps do
not) and the EPISODE MODEL for locked jump targets (plain jumps:
ignored + logged; dynamic episode jumps: pending until the
Diagnostiker's EpisodeDecision, then re-ask / same-episode-ignore;
predeclared: deterministic re-ask) — plus the online criterion-update
proposals with the decided guard.

The navigator is pure and proposal-based: it never mutates the run;
the tests drive the run through the state machine (step 2.2) exactly
the way the Supervisor would apply the proposals.

All tests are deterministic and need no network access.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pydantic import ValidationError

from contracts import (
    CriterionStatus,
    QuestionState,
    RunStatus,
    ValueReading,
    new_run_id,
)
from interview_definition import (
    CriterionDefinition,
    EqRule,
    EpisodeConfiguration,
    InterviewDefinition,
    Jump,
    PastEpisodesConfig,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ValueDeclaration,
)
from interview_navigator import (
    EpisodeDecision,
    InterviewNavigator,
    NavigatorError,
)
from interview_run import InterviewRun

DEFINITION_VERSION = "navi_guide@v1"


def make_definition() -> InterviewDefinition:
    """
    A small v2 definition exercising the navigation features:

        sec_1
          qg_1.1  (criteria group: c_anh on v_mood)
            sq_1.1.1  jump -> sec_2 when c_anh SATISFIED
            sq_1.1.2
          sec_1.2           (nested section)
            qg_1.2.1
              sq_1.2.1  jumps: [-> sec_2 when c_anh, -> qg_2.1 always]
        sec_2
          qg_2.1
            sq_2.1.1  jump -> sq_1.1.2 (backward, unconditional)

    Postorder (document) order:
      sq_1.1.1, sq_1.1.2, sq_1.2.1, sq_2.1.1
    """
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="navi_guide",
        title="Navi Test Guide",
        values=[ValueDeclaration(name="v_mood", type="bool")],
        sections=[
            SectionGroup(
                id="sec_1",
                title="Mood",
                children=[
                    QuestionGroup(
                        id="qg_1.1",
                        title="Core mood",
                        strategy="all_required",
                        criteria_group=True,
                        questions=[
                            QuestionDefinition(
                                id="sq_1.1.1",
                                text="Mood low?",
                                jumps=[
                                    Jump(
                                        to="sec_2", on_criterion="c_anh"
                                    ),
                                ],
                            ),
                            QuestionDefinition(
                                id="sq_1.1.2", text="Sleep bad?"
                            ),
                        ],
                        criteria=[
                            CriterionDefinition(
                                id="c_anh",
                                rule=EqRule(value="v_mood", equals=True),
                            ),
                        ],
                    ),
                    SectionGroup(
                        id="sec_1.2",
                        title="Appetite",
                        children=[
                            QuestionGroup(
                                id="qg_1.2.1",
                                title="Appetite",
                                questions=[
                                    QuestionDefinition(
                                        id="sq_1.2.1",
                                        text="Appetite low?",
                                        jumps=[
                                            Jump(
                                                to="sec_2",
                                                on_criterion="c_anh",
                                            ),
                                            Jump(to="qg_2.1"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            SectionGroup(
                id="sec_2",
                title="Severity",
                children=[
                    QuestionGroup(
                        id="qg_2.1",
                        title="Severity",
                        questions=[
                            QuestionDefinition(
                                id="sq_2.1.1",
                                text="Severity high?",
                                jumps=[Jump(to="sq_1.1.2")],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_run() -> InterviewRun:
    return InterviewRun(make_definition(), new_run_id())


def nav(run: InterviewRun) -> InterviewNavigator:
    return InterviewNavigator(run.definition)


def close_question(run: InterviewRun, question_id: str) -> None:
    """UNSEEN -> ... -> CLOSED (the full happy path)."""
    run.require_question(question_id, "drive")
    run.reserve_question(question_id, "drive")
    run.deliver_question(question_id, "drive")
    run.record_answer(question_id, "drive")
    run.close_question(question_id, "drive")


def in_flight_question(
    run: InterviewRun, question_id: str, target: QuestionState
) -> None:
    """UNSEEN -> target (REQUIRED/RESERVED/DELIVERED/ANSWER_RECORDED/REVISED)."""
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
    run.revise_question(question_id, "drive")
    assert target is QuestionState.REVISED


def make_episode_definition(
    declaration: str = "dynamic",
) -> InterviewDefinition:
    """
    The episode-model definition (design addendum 2026-10-26): the
    same structure as make_definition, but the episodes block is
    enabled, qg_1.1 is marked episode_evaluable, and sq_1.1.2 carries
    an EPISODE JUMP that loops back to qg_1.1 (gate: c_anh
    NOT_SATISFIED):
      * dynamic mode    -> episode_jump: true (the clinician declares
                           the episode at run time),
      * predeclared mode-> episode: "e_one".
    """
    if declaration == "dynamic":
        episodes = EpisodeConfiguration(
            enabled=True, past=PastEpisodesConfig(declaration="dynamic")
        )
        episode_jump = Jump(
            to="qg_1.1",
            on_criterion="c_anh",
            when=CriterionStatus.NOT_SATISFIED,
            episode_jump=True,
        )
    else:
        episodes = EpisodeConfiguration(
            enabled=True,
            past=PastEpisodesConfig(
                declaration="predeclared", episodes=["e_one"]
            ),
        )
        episode_jump = Jump(
            to="qg_1.1",
            on_criterion="c_anh",
            when=CriterionStatus.NOT_SATISFIED,
            episode="e_one",
        )
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="navi_guide",
        title="Navi Test Guide",
        values=[ValueDeclaration(name="v_mood", type="bool")],
        sections=[
            SectionGroup(
                id="sec_1",
                title="Mood",
                children=[
                    QuestionGroup(
                        id="qg_1.1",
                        title="Core mood",
                        strategy="all_required",
                        criteria_group=True,
                        episode_evaluable=True,
                        questions=[
                            QuestionDefinition(
                                id="sq_1.1.1",
                                text="Mood low?",
                                jumps=[
                                    Jump(
                                        to="sec_2", on_criterion="c_anh"
                                    ),
                                ],
                            ),
                            QuestionDefinition(
                                id="sq_1.1.2",
                                text="Sleep bad?",
                                jumps=[episode_jump],
                            ),
                        ],
                        criteria=[
                            CriterionDefinition(
                                id="c_anh",
                                rule=EqRule(value="v_mood", equals=True),
                            ),
                        ],
                    ),
                    SectionGroup(
                        id="sec_1.2",
                        title="Appetite",
                        children=[
                            QuestionGroup(
                                id="qg_1.2.1",
                                title="Appetite",
                                questions=[
                                    QuestionDefinition(
                                        id="sq_1.2.1",
                                        text="Appetite low?",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            SectionGroup(
                id="sec_2",
                title="Severity",
                children=[
                    QuestionGroup(
                        id="qg_2.1",
                        title="Severity",
                        questions=[
                            QuestionDefinition(id="sq_2.1.1", text="Severity high?"),
                        ],
                    ),
                ],
            ),
        ],
        episodes=episodes,
    )


def make_episode_run(declaration: str = "dynamic") -> InterviewRun:
    return InterviewRun(make_episode_definition(declaration), new_run_id())


def make_optional_run() -> InterviewRun:
    """A group with one required + one optional question (the
    completion-time skip proposals)."""
    definition = InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="navi_guide",
        title="Navi Test Guide",
        values=[ValueDeclaration(name="v_mood", type="bool")],
        sections=[
            SectionGroup(
                id="sec_1",
                title="Mood",
                children=[
                    QuestionGroup(
                        id="qg_1.1",
                        title="Core mood",
                        strategy="all_required",
                        questions=[
                            QuestionDefinition(id="sq_1.1.1", text="Mood low?"),
                            QuestionDefinition(
                                id="sq_1.1.1.2",
                                text="More detail?",
                                required=False,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    return InterviewRun(definition, new_run_id())


ORDER = ["sq_1.1.1", "sq_1.1.2", "sq_1.2.1", "sq_2.1.1"]


class TestStructureAndOrder(unittest.TestCase):
    """The deterministic fallback order (postorder / document order)."""

    def test_postorder_order(self):
        # the nested section (sec_1.2) is traversed where it sits in
        # document order — before sec_2
        self.assertEqual(nav(make_run()).question_order, ORDER)

    def test_outstanding_and_open_fresh(self):
        run = make_run()
        self.assertEqual(nav(run).outstanding_questions(run), ORDER)
        self.assertEqual(nav(run).open_questions(run), ORDER)
        self.assertFalse(nav(run).is_complete(run))

    def test_complete_when_all_terminal(self):
        run = make_run()
        for q in ORDER:
            close_question(run, q)
        n = nav(run)
        self.assertEqual(n.outstanding_questions(run), [])
        self.assertEqual(n.open_questions(run), [])
        self.assertTrue(n.is_complete(run))

    def test_in_flight_questions_are_outstanding_not_open(self):
        run = make_run()
        in_flight_question(run, "sq_1.1.1", QuestionState.DELIVERED)
        n = nav(run)
        self.assertEqual(n.outstanding_questions(run), ORDER)
        self.assertEqual(n.open_questions(run), ORDER[1:])

    def test_required_questions_count_as_open(self):
        run = make_run()
        run.require_question("sq_1.1.1", "drive")
        self.assertIn("sq_1.1.1", nav(run).open_questions(run))


class TestCandidateSet(unittest.TestCase):
    """The candidate set (contracts.QuestionCandidateSet) handed to the
    interviewer agent."""

    def test_fresh_candidates_in_fallback_order(self):
        run = make_run()
        cs = nav(run).candidate_set(run, "turn_001")
        self.assertEqual(cs.run_id, run.run_id)
        self.assertEqual(cs.turn_id, "turn_001")
        self.assertFalse(cs.is_empty())
        self.assertEqual(
            [c.question_id for c in cs.candidates], ORDER)
        self.assertEqual(
            [c.ordering for c in cs.candidates], [0, 1, 2, 3])
        for i, cand in enumerate(cs.candidates):
            definition_question = run.definition.question_by_id(cand.question_id)
            self.assertEqual(cand.text, definition_question.text)
            self.assertIn(cand.group_id, ("qg_1.1", "qg_1.2.1", "qg_2.1"))
            self.assertIn(
                f"postorder position {i + 1}", cand.reason_required)

    def test_reserved_question_is_the_single_candidate(self):
        run = make_run()
        in_flight_question(run, "sq_1.1.1", QuestionState.RESERVED)
        cs = nav(run).candidate_set(run, "turn_002")
        self.assertEqual(len(cs.candidates), 1)
        self.assertEqual(cs.candidates[0].question_id, "sq_1.1.1")
        self.assertEqual(cs.candidates[0].ordering, 0)
        self.assertIn("RESERVED", cs.candidates[0].reason_required)

    def test_in_flight_has_no_candidates(self):
        for target in (QuestionState.DELIVERED,
                       QuestionState.ANSWER_RECORDED):
            with self.subTest(target=target):
                run = make_run()
                in_flight_question(run, "sq_1.1.1", target)
                cs = nav(run).candidate_set(run, "turn_001")
                self.assertEqual(cs.candidates, [])
                self.assertTrue(cs.is_empty())
                self.assertIsNone(nav(run).next_question(run))

    def test_revised_comes_first(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        in_flight_question(run, "sq_1.1.2", QuestionState.REVISED)
        run.require_question("sq_1.2.1", "drive")
        run.reserve_question("sq_1.2.1", "drive")
        cs = nav(run).candidate_set(run, "turn_003")
        self.assertEqual(
            [c.question_id for c in cs.candidates],
            ["sq_1.1.2", "sq_1.2.1"])
        self.assertEqual(cs.candidates[0].ordering, 0)
        self.assertIn("REVISED", cs.candidates[0].reason_required)

    def test_terminal_run_has_empty_candidate_set(self):
        run = make_run()
        for q in ORDER:
            close_question(run, q)
        run.complete_run("drive")
        self.assertEqual(run.status, RunStatus.COMPLETED)
        cs = nav(run).candidate_set(run, "turn_001")
        self.assertEqual(cs.candidates, [])
        self.assertTrue(cs.is_empty())
        self.assertIsNone(nav(run).next_question(run))

    def test_turn_id_must_match_the_run_scheme(self):
        run = make_run()
        with self.assertRaises(ValidationError):
            nav(run).candidate_set(run, "bogus")


class TestNextQuestion(unittest.TestCase):
    """The deterministic fallback next question (ordering 0)."""

    def test_walks_the_postorder(self):
        run = make_run()
        n = nav(run)
        for q in ORDER:
            self.assertEqual(n.next_question(run), q)
            close_question(run, q)
        self.assertIsNone(n.next_question(run))

    def test_none_while_answer_in_flight(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        in_flight_question(run, "sq_1.1.2", QuestionState.DELIVERED)
        self.assertIsNone(nav(run).next_question(run))

    def test_skip_still_moves_forward(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        run.require_question("sq_1.1.2", "drive")
        run.skip_question("sq_1.1.2", "drive")
        self.assertEqual(nav(run).next_question(run), "sq_1.2.1")
        run = make_run()
        run.require_question("sq_1.1.1", "drive")
        self.assertIn("sq_1.1.1", nav(run).open_questions(run))


class TestGroupCompletion(unittest.TestCase):
    """
    complete_group / group_is_complete: the group consequences (the
    jump deferral, owner decision 2026-10-26).
    """

    def test_group_is_complete_only_when_required_are_terminal(self):
        run = make_run()
        n = nav(run)
        self.assertFalse(n.group_is_complete(run, "qg_1.1"))
        close_question(run, "sq_1.1.1")
        self.assertFalse(n.group_is_complete(run, "qg_1.1"))
        run.require_question("sq_1.1.2", "drive")
        run.skip_question("sq_1.1.2", "drive")  # SKIPPED counts as terminal
        self.assertTrue(n.group_is_complete(run, "qg_1.1"))

    def test_complete_group_raises_while_incomplete(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        with self.assertRaises(NavigatorError):
            nav(run).complete_group(run, "qg_1.1")

    def test_unknown_group_fails_loudly(self):
        run = make_run()
        with self.assertRaises(NavigatorError):
            nav(run).group_is_complete(run, "qg_nope")
        with self.assertRaises(NavigatorError):
            nav(run).complete_group(run, "qg_nope")

    def test_no_fire_continues_postorder(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        gc = nav(run).complete_group(run, "qg_1.1")
        self.assertEqual(gc.group_id, "qg_1.1")
        self.assertEqual(gc.optional_skip_proposals, ())
        self.assertFalse(gc.resolution.fired)
        self.assertEqual(gc.resolution.next_question, "sq_1.2.1")
        self.assertEqual(gc.resolution.skip_proposals, ())

    def test_gate_is_evaluated_at_completion_time(self):
        # the criterion is read when the group BECOMES complete — not
        # when the group was entered: c_anh is decided online in the
        # middle of the group and the jump only fires at completion
        run = make_run()
        close_question(run, "sq_1.1.1")  # c_anh still OPEN
        run.set_criterion(
            "c_anh", CriterionStatus.SATISFIED, "decided online"
        )
        close_question(run, "sq_1.1.2")
        gc = nav(run).complete_group(run, "qg_1.1")
        self.assertTrue(gc.resolution.fired)
        self.assertEqual(gc.resolution.jump.to, "sec_2")
        self.assertEqual(gc.resolution.next_question, "sq_2.1.1")
        # the open questions bypassed by the forward jump:
        self.assertEqual(gc.resolution.skip_proposals, ("sq_1.2.1",))

    def test_optional_questions_are_skipped_at_completion(self):
        # qg_1.1: required sq_1.1.1 + optional sq_1.1.1.2. When the
        # required question becomes terminal the group is complete and
        # the still-open optional question is the skip proposal:
        run = make_optional_run()
        close_question(run, "sq_1.1.1")
        self.assertTrue(nav(run).group_is_complete(run, "qg_1.1"))
        gc = nav(run).complete_group(run, "qg_1.1")
        self.assertEqual(gc.optional_skip_proposals, ("sq_1.1.1.2",))
        self.assertFalse(gc.resolution.fired)
        # after the Supervisor applied the skip proposal:
        run.require_question("sq_1.1.1.2", "skip proposal")
        run.skip_question("sq_1.1.1.2", "skip proposal")
        self.assertEqual(
            run.question_state("sq_1.1.1.2"), QuestionState.SKIPPED
        )

    def test_resolve_jump_mid_group_raises(self):
        run = make_run()
        run.require_question("sq_1.1.1", "drive")  # in flight
        # sq_1.1.2 is still UNSEEN: qg_1.1 is not complete
        with self.assertRaises(NavigatorError):
            nav(run).resolve_jump(run, "sq_1.1.1")
        with self.assertRaises(NavigatorError):
            nav(run).resolve_jump(run, "no_such_question")


class TestJumpResolution(unittest.TestCase):
    """
    resolve_jump (completion-time primitive): the first declared fired
    jump wins; forward jumps propose skips for the bypassed open
    questions; backward jumps do not; a locked target is decided by
    the episode model (TestEpisodeLoopBack).
    """

    def test_no_jumps_declared(self):
        # sq_1.1.2 has no jumps; group qg_1.1 is complete:
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        res = nav(run).resolve_jump(run, "sq_1.1.2")
        self.assertFalse(res.fired)
        self.assertIsNone(res.jump)
        self.assertEqual(res.next_question, "sq_1.2.1")  # plain postorder
        self.assertEqual(res.skip_proposals, ())

    def test_gated_jump_not_fired_when_criterion_open(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        # c_anh is still OPEN -> the jump does not fire
        res = nav(run).resolve_jump(run, "sq_1.1.1")
        self.assertFalse(res.fired)
        self.assertEqual(res.next_question, "sq_1.2.1")
        self.assertEqual(res.skip_proposals, ())

    def test_forward_jump_fires_and_skips_bypassed(self):
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        run.set_criterion("c_anh", CriterionStatus.SATISFIED, "drive")
        res = nav(run).resolve_jump(run, "sq_1.1.1")
        self.assertTrue(res.fired)
        self.assertEqual(res.jump.to, "sec_2")
        self.assertEqual(res.target_id, "sec_2")
        # the first open question of the target subtree:
        self.assertEqual(res.next_question, "sq_2.1.1")
        # the open questions strictly between the source (pos 0) and the
        # target start (pos 3) are bypassed (sq_1.1.2 is closed):
        self.assertEqual(res.skip_proposals, ("sq_1.2.1",))
        self.assertEqual(res.reopen_proposals, ())
        self.assertIsNone(res.loop_note)
        self.assertIsNone(res.reframe_context)

    def test_first_declared_fired_jump_wins(self):
        # sq_1.2.1: [-> sec_2 when c_anh, -> qg_2.1 always]
        run = make_run()
        close_question(run, "sq_1.2.1")
        run.set_criterion("c_anh", CriterionStatus.SATISFIED, "drive")
        res = nav(run).resolve_jump(run, "sq_1.2.1")
        self.assertTrue(res.fired)
        self.assertEqual(res.jump.to, "sec_2")  # the FIRST jump
        # c_anh still OPEN: first does not fire, the unconditional one
        run2 = make_run()
        close_question(run2, "sq_1.2.1")
        res2 = nav(run2).resolve_jump(run2, "sq_1.2.1")
        self.assertTrue(res2.fired)
        self.assertEqual(res2.jump.to, "qg_2.1")
        self.assertEqual(res2.next_question, "sq_2.1.1")
        self.assertEqual(res2.skip_proposals, ())  # target is the last group

    def test_backward_jump_does_not_skip(self):
        # sq_2.1.1 answered last and jumps back; sq_1.1.2 is still open
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_2.1.1")
        res = nav(run).resolve_jump(run, "sq_2.1.1")
        self.assertTrue(res.fired)
        self.assertEqual(res.jump.to, "sq_1.1.2")
        self.assertEqual(res.next_question, "sq_1.1.2")
        self.assertEqual(res.skip_proposals, ())  # backward: no bypass

    def test_first_fired_jump_across_the_group_questions(self):
        # qg_1.1: sq_1.1.1 jumps -> sec_2 when c_anh SATISFIED;
        # sq_1.1.2's episode jump gates on NOT_SATISFIED. The
        # (postorder) first question whose jump fires wins:
        run = make_episode_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        run.set_criterion("c_anh", CriterionStatus.SATISFIED, "drive")
        gc = nav(run).complete_group(run, "qg_1.1")
        self.assertEqual(gc.resolution.jump.to, "sec_2")


class TestEpisodeLoopBack(unittest.TestCase):
    """
    The episode model (design addendum 2026-10-26): a jump that loops
    back into an already-assessed (LOCKED) target. A plain jump is
    ignored + logged; an episode jump re-asks the target FOR THAT
    EPISODE (dynamic: via the Diagnostiker's EpisodeDecision;
    predeclared: deterministically) — and the same question is never
    re-asked for the same real episode.
    """

    def _run_episode_jump_fired(self, run: InterviewRun) -> None:
        """
        qg_1.1 worked through for the now-episode, c_anh NOT_SATISFIED
        -> at completion the episode jump of sq_1.1.2 fires on a
        locked target.
        """
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        run.set_criterion(
            "c_anh", CriterionStatus.NOT_SATISFIED, "decided online"
        )
        assert nav(run).group_is_complete(run, "qg_1.1")

    def test_plain_jump_to_locked_target_is_ignored_with_note(self):
        # plain definition (no episodes block): the backward jump of
        # sq_2.1.1 to the locked sq_1.1.2 is ignored + logged
        run = make_run()
        close_question(run, "sq_1.1.1")
        close_question(run, "sq_1.1.2")
        close_question(run, "sq_2.1.1")
        res = nav(run).resolve_jump(run, "sq_2.1.1")
        self.assertTrue(res.fired)
        self.assertEqual(res.target_id, "sq_1.1.2")
        self.assertIsNone(res.reframe_context)
        self.assertIsNone(res.episode)
        self.assertFalse(res.pending_episode)
        self.assertEqual(res.reopen_proposals, ())
        self.assertIn("no episode aspect", res.loop_note)
        # the jump is ignored; the postorder continues:
        self.assertEqual(res.next_question, "sq_1.2.1")
        # the locked question is not re-asked and not skipped:
        self.assertNotIn("sq_1.1.2", res.skip_proposals)

    def test_dynamic_episode_jump_is_pending_until_the_decision(self):
        run = make_episode_run()
        self._run_episode_jump_fired(run)
        gc = nav(run).complete_group(run, "qg_1.1")
        res = gc.resolution
        self.assertTrue(res.fired)
        self.assertEqual(res.jump.to, "qg_1.1")
        self.assertTrue(res.pending_episode)
        self.assertIsNone(res.next_question)
        self.assertIn("Diagnostiker", res.loop_note)
        # a direct resolve_jump call gives the same pending resolution
        # (the group is complete — the call is legal):
        res2 = nav(run).resolve_jump(run, "sq_1.1.2")
        self.assertTrue(res2.pending_episode)

    def test_dynamic_new_episode_reasks_the_locked_target(self):
        run = make_episode_run()
        self._run_episode_jump_fired(run)
        # the clinician declares a new episode, the patient frames it:
        run.declare_episode(
            "another period", "clinician: there was another time"
        )
        run.record_episode_alias(
            "ep_1", "march 2026", "the patient specifies"
        )
        res = nav(run).resolve_jump(
            run,
            "sq_1.1.2",
            episode_decision=EpisodeDecision("ep_1", "new_episode"),
        )
        self.assertTrue(res.fired)
        self.assertFalse(res.pending_episode)
        self.assertEqual(res.episode, "ep_1")
        self.assertEqual(res.reopen_proposals, ("sq_1.1.1", "sq_1.1.2"))
        self.assertEqual(res.next_question, "sq_1.1.1")
        self.assertIsNone(res.loop_note)
        # the reframing names the new episode and the old one:
        self.assertIn("march 2026", res.reframe_context)
        self.assertIn("now-episode", res.reframe_context)
        # the Supervisor applies the reopen proposals:
        for qid in res.reopen_proposals:
            run.reopen_question(
                qid, "ep_1", "episode jump re-asks the target"
            )
        self.assertEqual(
            run.question_state("sq_1.1.1"), QuestionState.REQUIRED
        )
        self.assertEqual(run.question_episodes["sq_1.1.1"], ["now", "ep_1"])
        # the candidate set carries the episode context:
        cs = nav(run).candidate_set(run, "turn_003")
        self.assertIn("ep_1", cs.candidates[0].reason_required)

    def test_dynamic_same_episode_is_ignored(self):
        run = make_episode_run()
        self._run_episode_jump_fired(run)
        run.declare_episode("another period", "clinician")
        run.record_episode_alias("ep_1", "march 2026", "the patient")
        # a second declaration turns out to be the SAME real episode:
        run.declare_episode("the spring period", "clinician")
        run.record_episode_alias(
            "ep_2", "the month before uni", "the patient"
        )
        run.merge_episodes("ep_2", "ep_1", "Diagnostiker: identical")
        res = nav(run).resolve_jump(
            run,
            "sq_1.1.2",
            episode_decision=EpisodeDecision("ep_2", "same_episode"),
        )
        self.assertTrue(res.fired)
        self.assertFalse(res.pending_episode)
        self.assertEqual(res.episode, "ep_1")  # canonical after the merge
        self.assertEqual(res.reopen_proposals, ())
        self.assertIsNone(res.reframe_context)
        # ignored + logged, with the winner's alias as reference:
        self.assertIn("ep_1", res.loop_note)
        self.assertIn("march 2026", res.loop_note)
        # the postorder continues:
        self.assertEqual(res.next_question, "sq_1.2.1")

    def test_decision_with_unknown_episode_fails_loudly(self):
        run = make_episode_run()
        self._run_episode_jump_fired(run)
        with self.assertRaises(NavigatorError):
            nav(run).resolve_jump(
                run,
                "sq_1.1.2",
                episode_decision=EpisodeDecision("ep_9", "new_episode"),
            )

    def test_predeclared_episode_jump_reasks_and_then_is_ignored(self):
        run = make_episode_run("predeclared")
        self._run_episode_jump_fired(run)
        gc = nav(run).complete_group(run, "qg_1.1")
        res = gc.resolution
        self.assertTrue(res.fired)
        self.assertFalse(res.pending_episode)  # deterministic: no decision
        self.assertEqual(res.episode, "e_one")
        self.assertEqual(res.reopen_proposals, ("sq_1.1.1", "sq_1.1.2"))
        self.assertIn("e_one", res.reframe_context)
        # the Supervisor applies the reopen proposals and works the
        # group through for e_one again:
        for qid in res.reopen_proposals:
            run.reopen_question(qid, "e_one", "episode jump")
        for qid in ("sq_1.1.1", "sq_1.1.2"):
            run.reserve_question(qid, "nav")
            run.deliver_question(qid, "output")
            run.record_answer(qid, "answered for e_one")
            run.close_question(qid, "closed for e_one")
        # the jump fires again: the target is assessed for e_one now ->
        # ignored + logged (the same question is not re-asked for the
        # same episode)
        res2 = nav(run).resolve_jump(run, "sq_1.1.2")
        self.assertEqual(res2.episode, "e_one")
        self.assertEqual(res2.reopen_proposals, ())
        self.assertIsNone(res2.reframe_context)
        self.assertIn("already assessed for episode 'e_one'", res2.loop_note)


class TestCriterionUpdates(unittest.TestCase):
    """propose_criterion_updates: online evaluation + the decided guard."""

    def test_no_change_with_no_readings(self):
        run = make_run()
        self.assertEqual(nav(run).propose_criterion_updates(run, {}), {})

    def test_newly_decided_criteria_are_proposed(self):
        run = make_run()
        readings = {"v_mood": ValueReading(name="v_mood", value=True)}
        # c_anh: v_mood == True -> SATISFIED
        self.assertEqual(
            nav(run).propose_criterion_updates(run, readings),
            {"c_anh": CriterionStatus.SATISFIED},
        )

    def test_unclear_reading_is_proposed(self):
        run = make_run()
        readings = {
            "v_mood": ValueReading(name="v_mood", value=None,
                                   reliable=False)
        }
        self.assertEqual(
            nav(run).propose_criterion_updates(run, readings),
            {"c_anh": CriterionStatus.UNCLEAR},
        )

    def test_decided_guard_no_re_evaluation(self):
        run = make_run()
        run.set_criterion("c_anh", CriterionStatus.SATISFIED, "drive")
        # the reading now says the opposite — but c_anh is decided and
        # must NOT be re-evaluated
        readings = {"v_mood": ValueReading(name="v_mood", value=False)}
        self.assertEqual(
            nav(run).propose_criterion_updates(run, readings), {})

    def test_unclear_reopens_a_decided_criterion(self):
        run = make_run()
        run.set_criterion("c_anh", CriterionStatus.SATISFIED, "drive")
        # a clarification shows the answer was misread -> UNCLEAR (legal
        # transition decided -> UNCLEAR); the criterion is re-evaluated
        run.set_criterion("c_anh", CriterionStatus.UNCLEAR, "drive")
        readings = {"v_mood": ValueReading(name="v_mood", value=True)}
        self.assertEqual(
            nav(run).propose_criterion_updates(run, readings),
            {"c_anh": CriterionStatus.SATISFIED},
        )


if __name__ == "__main__":
    unittest.main()