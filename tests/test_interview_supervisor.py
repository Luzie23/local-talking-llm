"""
Unit tests for backend/interview_supervisor.py (RECONSTRUCTION_PLAN.md
Phase 2, step 2.5): the deterministic Supervisor — the SOLE writer of
the InterviewRun state machine and the applier of the Navigator's
proposals. The tests cover the atomic question flow (require /
reserve / deliver / apply_answer; the answer readings stored in the
ACTIVE episode context, OFQ-2; the online criterion proposals
applied), the GROUP-COMPLETION consequences (deferred jumps resolved
once per (group, episode context): skip proposals, the OUT-OF-SCOPE
exit aborting the run, OFQ-6), the EPISODE re-entry (OFQ-1: the auto
policy deciding pending dynamic episode jumps inline; the LLM-driven
mode returning PENDING_EPISODE and applying the Diagnostiker's
EpisodeDecision via apply_episode_decision; the re-entry re-asking
the locked target for the episode with the mood bootstrap; the
same-episode ignore; the max_past_episodes bound), the SCID A52
ESCALATION force-coding (OFQ-7) and the {text token} rendering
(OFQ-4/OFQ-5) via next_question_text.

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
    ValueReading,
    new_run_id,
)
from interview_definition import (
    CriterionDefinition,
    CriterionScope,
    EqRule,
    EpisodeConfiguration,
    ForcedCriterion,
    InterviewDefinition,
    Jump,
    PastEpisodesConfig,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ValueDeclaration,
)
from interview_navigator import EpisodeDecision, NavigatorError
from interview_supervisor import InterviewSupervisor, SupervisorOutcome

DEFINITION_VERSION = "sup_guide@v1"


def make_basic_definition() -> InterviewDefinition:
    """
    A small v2 definition exercising the navigation features (the
    navigator test's fixture, episodes disabled):

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
        guide_id="sup_guide",
        title="Sup Test Guide",
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


def make_episode_definition(
    declaration: str = "dynamic",
) -> InterviewDefinition:
    """
    The episode-model definition: the basic structure, but the
    episodes block is enabled, qg_1.1 is marked episode_evaluable,
    sq_1.1.2 carries an EPISODE JUMP that loops back to qg_1.1 (gate:
    c_anh NOT_SATISFIED), and the mood values (the re-entry mood
    bootstrap's inputs/outputs) are declared:
      * dynamic mode    -> episode_jump: true,
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
        guide_id="sup_guide",
        title="Sup Test Guide",
        values=[
            ValueDeclaration(name="v_mood", type="bool"),
            ValueDeclaration(name="past_elevated_mood", type="bool"),
            ValueDeclaration(name="past_irritable_mood", type="bool"),
            ValueDeclaration(name="episode_elevated_mood", type="bool"),
            ValueDeclaration(name="episode_irritable_mood", type="bool"),
        ],
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
                                # episode-scoped (OFQ-2): the criterion
                                # is decided per episode context, so
                                # the re-entered battery gets its own
                                # decision.
                                scope=CriterionScope(episode="current"),
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
                            QuestionDefinition(
                                id="sq_2.1.1", text="Severity high?"
                            ),
                        ],
                    ),
                ],
            ),
        ],
        episodes=episodes,
    )


def make_oos_definition() -> InterviewDefinition:
    """
    A definition whose only jump exits OUTSIDE the definition (OFQ-6):
    sq_1.1.1 jumps to the free-form reference 'external_B' (a manual
    section of the full SCID, not part of this excerpt).
    """
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="sup_guide",
        title="Sup Test Guide",
        values=[],
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
                            QuestionDefinition(
                                id="sq_1.1.1",
                                text="Mood low?",
                                jumps=[
                                    Jump(
                                        to="external_B",
                                        out_of_scope=True,
                                        note="continue with SCID section B "
                                        "(outside this excerpt)",
                                    ),
                                ],
                            ),
                            QuestionDefinition(
                                id="sq_1.1.2", text="Sleep bad?"
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_escalation_definition() -> InterviewDefinition:
    """
    The SCID A52 escalation structure (OFQ-7): the duration group
    qg_a30 force-jumps to the severity group qg_a39 when the duration
    criterion is SATISFIED; the escalation force-codes the duration
    criterion and the symptom threshold.
    """
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="sup_guide",
        title="Sup Test Guide",
        values=[
            ValueDeclaration(name="v_duration", type="bool"),
            ValueDeclaration(name="v_symptoms", type="bool"),
        ],
        sections=[
            SectionGroup(
                id="sec_a",
                title="Mania",
                children=[
                    QuestionGroup(
                        id="qg_a30",
                        title="Duration",
                        strategy="all_required",
                        criteria_group=True,
                        questions=[
                            QuestionDefinition(
                                id="sq_a30_1",
                                text="Mania lasted weeks?",
                                jumps=[
                                    Jump(
                                        to="qg_a39",
                                        on_criterion="c_a30_week",
                                        force_criterion=[
                                            ForcedCriterion(
                                                criterion="c_a30_week"
                                            ),
                                            ForcedCriterion(
                                                criterion="c_mania_symptom_threshold"
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                        criteria=[
                            CriterionDefinition(
                                id="c_a30_week",
                                rule=EqRule(value="v_duration", equals=True),
                            ),
                            CriterionDefinition(
                                id="c_mania_symptom_threshold",
                                rule=EqRule(value="v_symptoms", equals=True),
                            ),
                        ],
                    ),
                    QuestionGroup(
                        id="qg_a39",
                        title="Severity",
                        questions=[
                            QuestionDefinition(
                                id="sq_a39_1", text="Severity high?"
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_token_definition() -> InterviewDefinition:
    """One question whose text carries both {text tokens} (OFQ-4/OFQ-5)."""
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="sup_guide",
        title="Sup Test Guide",
        values=[
            ValueDeclaration(name="episode_elevated_mood", type="bool"),
            ValueDeclaration(name="episode_irritable_mood", type="bool"),
        ],
        sections=[
            SectionGroup(
                id="sec_1",
                title="Mood",
                children=[
                    QuestionGroup(
                        id="qg_1.1",
                        title="Mood",
                        questions=[
                            QuestionDefinition(
                                id="sq_1.1.1",
                                text=(
                                    "Gibt es {other_time} Zeit, in der "
                                    "Sie sich {symptom_term} gefühlt haben?"
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def make_optional_definition() -> InterviewDefinition:
    """A group with one required + one optional question (the
    completion-time skip proposals)."""
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="sup_guide",
        title="Sup Test Guide",
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


def readings(**values) -> list[ValueReading]:
    """The Diagnostiker's value readings for one answer."""
    return [
        ValueReading(name=name, value=value, reliable=True)
        for name, value in values.items()
    ]


def ask_and_answer(
    sup: InterviewSupervisor, expected_question: str, **values
) -> SupervisorOutcome:
    """
    The atomic flow for one question: reserve the expected next
    question, deliver it, apply the given answer readings.
    """
    outcome = sup.require_and_reserve_next()
    assert outcome.kind == SupervisorOutcome.CONTINUE, outcome
    assert outcome.next_question == expected_question, (
        outcome,
        expected_question,
    )
    sup.deliver()
    return sup.apply_answer(expected_question, readings(**values))


class TestOutOfScope(unittest.TestCase):
    """
    The OUT-OF-SCOPE exit (OFQ-6): a jump whose continuation target
    lies outside the loaded definition is an explicit, logged ABORT —
    never a silently swallowed jump.
    """

    def test_out_of_scope_jump_aborts_the_run(self):
        sup = InterviewSupervisor(make_oos_definition(), new_run_id())
        out = ask_and_answer(sup, "sq_1.1.1")
        self.assertEqual(out.next_question, "sq_1.1.2")
        # qg_1.1 complete: the unconditional out-of-scope jump fires.
        out = ask_and_answer(sup, "sq_1.1.2")
        self.assertEqual(out.kind, SupervisorOutcome.OUT_OF_SCOPE)
        self.assertIn("external_B", out.message)
        self.assertIn("SCID section B", out.message)
        self.assertEqual(sup.status, RunStatus.ABORTED)
        self.assertIsNotNone(sup.out_of_scope)
        self.assertEqual(sup.out_of_scope.target_ref, "external_B")
        self.assertIn("SCID section B", sup.out_of_scope.note)
        # the run is over: no further steps are possible.
        with self.assertRaises(NavigatorError):
            sup.require_and_reserve_next()

    def test_out_of_scope_exit_is_recorded_once(self):
        sup = InterviewSupervisor(make_oos_definition(), new_run_id())
        ask_and_answer(sup, "sq_1.1.1")
        ask_and_answer(sup, "sq_1.1.2")
        self.assertEqual(sup.out_of_scope.target_ref, "external_B")
        # the cause log ends with the explicit out-of-scope abort:
        causes = [entry.cause for entry in sup.run.transitions]
        self.assertTrue(
            any("out-of-scope exit" in cause for cause in causes),
            causes,
        )


class TestQuestionFlow(unittest.TestCase):
    """The atomic flow: require / reserve / deliver / apply_answer."""

    def test_walks_the_postorder_and_completes(self):
        sup = InterviewSupervisor(make_basic_definition(), new_run_id())
        self.assertEqual(sup.status, RunStatus.ACTIVE)
        self.assertEqual(sup.active_episode, "now")
        # v_mood=False: c_anh NOT_SATISFIED -> the sec_2 jump of
        # sq_1.1.1 does not fire; the plain postorder holds.
        out = ask_and_answer(sup, "sq_1.1.1", v_mood=False)
        self.assertEqual(out.kind, SupervisorOutcome.CONTINUE)
        self.assertEqual(out.next_question, "sq_1.1.2")
        self.assertEqual(
            sup.criterion_state("c_anh"), CriterionStatus.NOT_SATISFIED
        )
        out = ask_and_answer(sup, "sq_1.1.2")
        # qg_1.1 complete: no fired jump (c_anh NOT_SATISFIED) -> the
        # postorder continues with sq_1.2.1.
        self.assertEqual(out.next_question, "sq_1.2.1")
        # v_mood=True now — but c_anh is already decided
        # (NOT_SATISFIED): the DECIDED GUARD means no re-evaluation
        # (never decided -> decided), so the sec_2 jump of sq_1.2.1
        # does not fire and the UNCONDITIONAL jump -> qg_2.1 does:
        # the next question is sq_2.1.1.
        out = ask_and_answer(sup, "sq_1.2.1", v_mood=True)
        self.assertEqual(
            sup.criterion_state("c_anh"), CriterionStatus.NOT_SATISFIED
        )
        self.assertEqual(out.next_question, "sq_2.1.1")
        # sq_2.1.1's backward jump to the locked sq_1.1.2 is ignored
        # + logged; nothing more is to ask.
        out = ask_and_answer(sup, "sq_2.1.1")
        self.assertIsNone(out.next_question)
        out = sup.require_and_reserve_next()
        self.assertEqual(out.kind, SupervisorOutcome.COMPLETE)
        self.assertEqual(sup.status, RunStatus.COMPLETED)
        self.assertIsNone(sup.out_of_scope)

    def test_fired_forward_jump_skips_bypassed_questions(self):
        sup = InterviewSupervisor(make_basic_definition(), new_run_id())
        # v_mood=True: c_anh SATISFIED -> when qg_1.1 completes (after
        # sq_1.1.2), sq_1.1.1's jump to sec_2 fires; the still-open
        # sq_1.2.1 is bypassed (skipped) and the flow lands on sq_2.1.1.
        out = ask_and_answer(sup, "sq_1.1.1", v_mood=True)
        self.assertEqual(out.next_question, "sq_1.1.2")
        out = ask_and_answer(sup, "sq_1.1.2")
        self.assertEqual(out.next_question, "sq_2.1.1")
        self.assertEqual(
            sup.question_state("sq_1.2.1"), QuestionState.SKIPPED
        )
        self.assertEqual(sup.status, RunStatus.ACTIVE)

    def test_answer_in_flight_cannot_advance(self):
        sup = InterviewSupervisor(make_basic_definition(), new_run_id())
        out = sup.require_and_reserve_next()
        self.assertEqual(out.next_question, "sq_1.1.1")
        sup.deliver()
        # the answer is in flight: nothing can be reserved/asked...
        self.assertIsNone(sup.next_question_text())
        out = sup.require_and_reserve_next()
        self.assertEqual(out.kind, SupervisorOutcome.ANSWER_IN_FLIGHT)
        self.assertIsNone(out.next_question)
        # ...and the candidate set is empty:
        cs = sup.candidate_set("turn_001")
        self.assertEqual(cs.candidates, [])

    def test_apply_answer_requires_a_delivered_question(self):
        sup = InterviewSupervisor(make_basic_definition(), new_run_id())
        with self.assertRaises(NavigatorError):
            sup.apply_answer("sq_1.1.1", readings(v_mood=True))

    def test_apply_answer_after_completion_fails(self):
        sup = InterviewSupervisor(make_token_definition(), new_run_id())
        ask_and_answer(sup, "sq_1.1.1")
        out = sup.require_and_reserve_next()
        self.assertEqual(out.kind, SupervisorOutcome.COMPLETE)
        with self.assertRaises(NavigatorError):
            sup.apply_answer("sq_1.1.1", readings())

    def test_optional_questions_are_skipped_at_completion(self):
        sup = InterviewSupervisor(make_optional_definition(), new_run_id())
        out = ask_and_answer(sup, "sq_1.1.1", v_mood=True)
        # the group is complete: the optional question is skipped, and
        # there is nothing more to ask.
        self.assertEqual(
            sup.question_state("sq_1.1.1.2"), QuestionState.SKIPPED
        )
        self.assertIsNone(out.next_question)
        out = sup.require_and_reserve_next()
        self.assertEqual(out.kind, SupervisorOutcome.COMPLETE)


class TestEpisodeReentryAuto(unittest.TestCase):
    """
    The EPISODE re-entry (OFQ-1), atomic mode (auto_episode_decisions=
    True): a pending dynamic episode jump is decided by the
    Supervisor's deterministic auto policy INLINE — the locked target
    is re-asked FOR the episode (context switch, mood bootstrap,
    re-opens, reservation).
    """

    def _answer_first_pass_negative(self, sup: InterviewSupervisor):
        """
        Work qg_1.1 through for the now-episode with c_anh
        NOT_SATISFIED (v_mood=False): at completion the episode jump of
        sq_1.1.2 fires on the LOCKED target qg_1.1 -> pending -> the
        auto policy decides.
        """
        out = ask_and_answer(sup, "sq_1.1.1", v_mood=False)
        self.assertEqual(out.next_question, "sq_1.1.2")
        return ask_and_answer(sup, "sq_1.1.2")

    def test_auto_reentry_reasks_the_target_for_a_new_episode(self):
        sup = InterviewSupervisor(make_episode_definition(), new_run_id())
        out = self._answer_first_pass_negative(sup)
        # the auto policy: the target qg_1.1 is assessed for the
        # active episode 'now' -> a NEW past episode (ep_1) is
        # declared and re-entered:
        self.assertEqual(out.kind, SupervisorOutcome.CONTINUE)
        self.assertEqual(out.episode, "ep_1")
        self.assertEqual(out.next_question, "sq_1.1.1")
        self.assertEqual(sup.active_episode, "ep_1")
        self.assertIn("ep_1", sup.episodes)
        # the locked target's questions were re-opened FOR ep_1; the
        # first is reserved (the deterministic next question):
        self.assertEqual(
            sup.question_state("sq_1.1.1"), QuestionState.RESERVED
        )
        self.assertEqual(
            sup.question_state("sq_1.1.2"), QuestionState.REQUIRED
        )
        self.assertEqual(
            sup.run.question_episodes["sq_1.1.1"], ["now", "ep_1"]
        )
        # the candidate set carries the episode context:
        cs = sup.candidate_set("turn_003")
        self.assertIn("ep_1", cs.candidates[0].reason_required)

    def test_mood_bootstrap_seeds_the_reentered_episode(self):
        sup = InterviewSupervisor(make_episode_definition(), new_run_id())
        # the A54 screen (asked in the now context) pinned the PAST
        # episode's mood as irritable:
        sup.run.set_value_for(
            "now",
            ValueReading(name="past_irritable_mood", value=True),
            "A54 screen: irritable mood",
        )
        out = self._answer_first_pass_negative(sup)
        self.assertEqual(out.episode, "ep_1")
        # the re-entered episode's mood context was bootstrapped from
        # the screen: the {symptom_term} token now renders 'reizbar'.
        self.assertTrue(sup.value("episode_irritable_mood", "ep_1").value)
        self.assertIsNone(sup.value("episode_elevated_mood", "ep_1"))
        # the readings stay separated per context:
        self.assertIsNone(sup.value("episode_irritable_mood", "now"))
        self.assertIsNone(sup.value("past_irritable_mood", "ep_1"))

    def test_reentered_group_can_be_decided_and_the_flow_continues(self):
        sup = InterviewSupervisor(make_episode_definition(), new_run_id())
        out = self._answer_first_pass_negative(sup)
        self.assertEqual(out.next_question, "sq_1.1.1")
        self.assertEqual(sup.active_episode, "ep_1")
        # the re-asked battery is answered FOR ep_1: v_mood=True
        # decides c_anh SATISFIED IN THE EP_1 CONTEXT (the now context
        # keeps NOT_SATISFIED — episode-scoped criteria, OFQ-2):
        sup.deliver()
        out = sup.apply_answer("sq_1.1.1", readings(v_mood=True))
        self.assertEqual(
            sup.criterion_state("c_anh"), CriterionStatus.SATISFIED
        )
        self.assertEqual(
            sup.run.episode_criterion_states("now")["c_anh"],
            CriterionStatus.NOT_SATISFIED,
        )
        self.assertEqual(sup.value("v_mood", "now").value, False)
        self.assertEqual(sup.value("v_mood", "ep_1").value, True)
        self.assertEqual(out.next_question, "sq_1.1.2")
        # ep_1 pass complete: the episode jump (c_anh NOT_SATISFIED in
        # ep_1) does not fire; sq_1.1.1's plain jump -> sec_2 (c_anh
        # SATISFIED in ep_1) does: the bypassed sq_1.2.1 is skipped and
        # the flow lands on sq_2.1.1.
        sup.require_and_reserve_next()
        sup.deliver()
        out = sup.apply_answer("sq_1.1.2")
        self.assertEqual(out.next_question, "sq_2.1.1")
        self.assertEqual(
            sup.question_state("sq_1.2.1"), QuestionState.SKIPPED
        )
        # and the interview completes:
        ask_and_answer(sup, "sq_2.1.1")
        out = sup.require_and_reserve_next()
        self.assertEqual(out.kind, SupervisorOutcome.COMPLETE)
        self.assertEqual(sup.status, RunStatus.COMPLETED)

    def test_reentry_loop_is_bounded(self):
        sup = InterviewSupervisor(
            make_episode_definition(), new_run_id(), max_past_episodes=2
        )
        # every pass keeps c_anh NOT_SATISFIED in its episode context
        # -> every group completion declares a NEW past episode:
        out = self._answer_first_pass_negative(sup)
        self.assertEqual(out.episode, "ep_1")
        sup.deliver()
        out = sup.apply_answer("sq_1.1.1", readings(v_mood=False))
        self.assertEqual(out.next_question, "sq_1.1.2")
        out = ask_and_answer(sup, "sq_1.1.2")
        self.assertEqual(out.episode, "ep_2")
        # the third completion would need ep_3 — past the bound:
        sup.deliver()
        sup.apply_answer("sq_1.1.1", readings(v_mood=False))
        with self.assertRaises(NavigatorError):
            ask_and_answer(sup, "sq_1.1.2")

    def test_reenter_episode_api_rejects_unknown_episode(self):
        sup = InterviewSupervisor(make_episode_definition(), new_run_id())
        self._answer_first_pass_negative(sup)
        # after the auto re-entry, ep_1 is active; ep_2 was never
        # declared -> the context switch fails loudly:
        with self.assertRaises(ValueError):
            sup.reenter_episode("ep_2", "qg_1.1")


class TestEpisodeDecisionsLLMMode(unittest.TestCase):
    """
    The LLM-driven episode decision (OFQ-1, auto_episode_decisions=
    False): the Supervisor reports PENDING_EPISODE and the
    Diagnostiker's EpisodeDecision is applied via
    apply_episode_decision — the re-entry (new_episode) or the
    same-episode ignore.
    """

    def _pending(self, sup: InterviewSupervisor) -> SupervisorOutcome:
        ask_and_answer(sup, "sq_1.1.1", v_mood=False)
        return ask_and_answer(sup, "sq_1.1.2")

    def test_pending_episode_outcome(self):
        sup = InterviewSupervisor(
            make_episode_definition(),
            new_run_id(),
            auto_episode_decisions=False,
        )
        out = self._pending(sup)
        self.assertEqual(out.kind, SupervisorOutcome.PENDING_EPISODE)
        self.assertEqual(out.pending_source, "sq_1.1.2")
        # nothing was applied: the context is untouched and the target
        # questions are still terminal (not re-opened):
        self.assertEqual(sup.active_episode, "now")
        self.assertEqual(
            sup.question_state("sq_1.1.1"), QuestionState.CLOSED
        )
        self.assertEqual(sup.status, RunStatus.ACTIVE)

    def test_apply_new_episode_decision_reenters(self):
        sup = InterviewSupervisor(
            make_episode_definition(),
            new_run_id(),
            auto_episode_decisions=False,
        )
        out = self._pending(sup)
        self.assertEqual(out.kind, SupervisorOutcome.PENDING_EPISODE)
        # the clinician declares the episode, the patient frames it:
        sup.run.declare_episode(
            "another period", "clinician: there was another time"
        )
        sup.run.record_episode_alias(
            "ep_1", "march 2026", "the patient specifies"
        )
        out = sup.apply_episode_decision(
            "sq_1.1.2", EpisodeDecision("ep_1", "new_episode")
        )
        self.assertEqual(out.kind, SupervisorOutcome.CONTINUE)
        self.assertEqual(out.episode, "ep_1")
        self.assertEqual(out.next_question, "sq_1.1.1")
        self.assertEqual(sup.active_episode, "ep_1")
        self.assertEqual(
            sup.question_state("sq_1.1.1"), QuestionState.RESERVED
        )
        self.assertEqual(
            sup.run.question_episodes["sq_1.1.1"], ["now", "ep_1"]
        )

    def test_apply_same_episode_decision_is_ignored(self):
        sup = InterviewSupervisor(
            make_episode_definition(),
            new_run_id(),
            auto_episode_decisions=False,
        )
        self._pending(sup)
        sup.run.declare_episode("another period", "clinician")
        sup.run.record_episode_alias(
            "ep_1", "march 2026", "the patient"
        )
        # a second declaration turns out to be the SAME real episode:
        sup.run.declare_episode("the spring period", "clinician")
        sup.run.record_episode_alias(
            "ep_2", "the month before uni", "the patient"
        )
        sup.run.merge_episodes("ep_2", "ep_1", "Diagnostiker: identical")
        out = sup.apply_episode_decision(
            "sq_1.1.2", EpisodeDecision("ep_2", "same_episode")
        )
        # ignored + logged: nothing is re-asked, the postorder
        # continues (canonical episode 'ep_1' after the merge):
        self.assertEqual(out.kind, SupervisorOutcome.CONTINUE)
        self.assertEqual(out.episode, "ep_1")
        self.assertEqual(out.next_question, "sq_1.2.1")
        self.assertEqual(
            sup.question_state("sq_1.1.1"), QuestionState.CLOSED
        )
        self.assertEqual(sup.active_episode, "now")

    def test_decision_with_unknown_episode_fails_loudly(self):
        sup = InterviewSupervisor(
            make_episode_definition(),
            new_run_id(),
            auto_episode_decisions=False,
        )
        self._pending(sup)
        with self.assertRaises(NavigatorError):
            sup.apply_episode_decision(
                "sq_1.1.2", EpisodeDecision("ep_9", "new_episode")
            )


class TestEscalationCoding(unittest.TestCase):
    """
    The DECLARATIVE jump force-coding (OFQ-7): a jump that declares
    `force_criterion` and fires (target first-asked) force-codes its
    declared criteria in the active context — a coding decision, not a
    routing step. The test mirrors the SCID A52 escalation: the jump
    onto qg_a39 declares the mania duration criterion and the symptom
    threshold to be force-coded SATISFIED.
    """

    def test_escalation_force_codes_both_criteria(self):
        sup = InterviewSupervisor(
            make_escalation_definition(), new_run_id()
        )
        # weeks of mania (c_a30_week SATISFIED), but the symptom
        # threshold NOT met (v_symptoms=False -> NOT_SATISFIED):
        out = ask_and_answer(
            sup, "sq_a30_1", v_duration=True, v_symptoms=False
        )
        # the escalation jump fired (qg_a39 is the next question) and
        # BOTH criteria are now force-coded SATISFIED in the active
        # context:
        self.assertEqual(out.next_question, "sq_a39_1")
        self.assertEqual(
            sup.criterion_state("c_a30_week"), CriterionStatus.SATISFIED
        )
        self.assertEqual(
            sup.criterion_state("c_mania_symptom_threshold"),
            CriterionStatus.SATISFIED,
        )
        # the force-coding is logged on the run (c_a30_week was already
        # SATISFIED from the answer, so only the threshold — reopened via
        # UNCLEAR — leaves a force-coding record):
        causes = [entry.cause for entry in sup.run.transitions]
        self.assertTrue(
            any(
                "force-codes c_mania_symptom_threshold" in cause
                for cause in causes
            ),
            causes,
        )
        # the run is still active: the severity question follows.
        self.assertEqual(sup.status, RunStatus.ACTIVE)

    def test_escalation_not_forced_when_the_jump_does_not_fire(self):
        sup = InterviewSupervisor(
            make_escalation_definition(), new_run_id()
        )
        # no weeks of mania: the jump does not fire, nothing is
        # force-coded (the threshold criterion stays open/unclear).
        out = ask_and_answer(
            sup, "sq_a30_1", v_duration=False, v_symptoms=False
        )
        self.assertEqual(
            sup.criterion_state("c_a30_week"), CriterionStatus.NOT_SATISFIED
        )
        self.assertNotEqual(
            sup.criterion_state("c_mania_symptom_threshold"),
            CriterionStatus.SATISFIED,
        )
        # the plain postorder continues to the next group:
        self.assertEqual(out.next_question, "sq_a39_1")


class TestPlaceholders(unittest.TestCase):
    """
    The {text token} rendering (OFQ-4/OFQ-5): next_question_text
    renders {symptom_term} and {other_time} from the run's current state.
    """

    def test_default_rendering_before_the_mood_is_pinned(self):
        sup = InterviewSupervisor(make_token_definition(), new_run_id())
        self.assertEqual(
            sup.next_question_text(),
            "Gibt es eine Zeit, in der Sie sich euphorisch gefühlt "
            "haben?",
        )

    def test_symptom_term_follows_the_active_context(self):
        sup = InterviewSupervisor(make_token_definition(), new_run_id())
        sup.run.set_value_for(
            "now",
            ValueReading(name="episode_irritable_mood", value=True),
            "A54 screen: irritable mood",
        )
        self.assertEqual(
            sup.next_question_text(),
            "Gibt es eine andere Zeit, in der Sie sich reizbar "
            "gefühlt haben?",
        )

    def test_other_time_article_is_ep_aware(self):
        sup = InterviewSupervisor(make_token_definition(), new_run_id())
        # no established mood in the now-episode yet -> 'eine':
        self.assertIn("eine Zeit", sup.next_question_text())
        # the now-episode's mood is pinned -> 'eine andere':
        sup.run.set_value_for(
            "now",
            ValueReading(name="episode_elevated_mood", value=True),
            "A54 screen: elevated mood",
        )
        self.assertIn("eine andere Zeit", sup.next_question_text())


class TestRunAccessors(unittest.TestCase):
    """The Supervisor owns the run: ids, accessors, error paths."""

    def test_run_id_generated_and_kept(self):
        rid = new_run_id()
        sup = InterviewSupervisor(make_basic_definition(), rid)
        self.assertEqual(sup.run_id, rid)
        self.assertTrue(
            InterviewSupervisor(
                make_basic_definition()
            ).run_id.startswith("run_")
        )

    def test_deliver_without_reservation_fails(self):
        sup = InterviewSupervisor(make_basic_definition(), new_run_id())
        with self.assertRaises(NavigatorError):
            sup.deliver()

    def test_readings_reflect_the_active_context(self):
        sup = InterviewSupervisor(make_episode_definition(), new_run_id())
        sup.run.set_value_for(
            "now", ValueReading(name="v_mood", value=True), "test"
        )
        self.assertEqual(sup.value("v_mood").value, True)
        self.assertEqual(sup.value("v_mood", "now").value, True)
        # no past episode has readings yet:
        self.assertEqual(set(sup.readings_by_context()), {"now"})


if __name__ == "__main__":
    unittest.main()
