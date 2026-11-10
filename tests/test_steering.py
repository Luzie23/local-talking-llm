"""
Unit tests for the WInf-INSPIRED STEERING POLICY (RECONSTRUCTION_PLAN.md
Phase 2, step 2.5):

  * the deterministic command checks — normalize_command /
    is_explicit_skip (the explicit skip command, normalized exact
    match, no LLM) and the failure-choice continue/pause phrases;
  * the DETERMINISTIC STEERING REACTION TABLE (interview_navigator.
    steering_decision — a pure function of (intent, answer clarity,
    the per-kind re-ask counters, the SteeringConfig thresholds)):
    refusal -> the SKIP-REQUEST reaction (the question is probed
    first — ask why + alternatives — and skipped only at the
    skip-probe threshold; the explicit skip command shares the same
    decision via skip_request_decision), clarification_request ->
    rephrase re-ask (unlimited by default — does not consume the
    clarification budget), off_topic/unclear -> redirect then skip,
    cannot_answer -> gentle follow-up then skip, answer+clear ->
    proceed, answer+unclear -> clarification then skip; the per-kind
    counters are independent of each other;
  * the PIPELINE integration (the Supervisor with scripted fakes):
    the skip-REQUEST pre-checks run BEFORE the Diagnostiker (no LLM
    call) and probe the question (DELIVERED -> REVISED -> DELIVERED,
    FOLLOWUP with the pre-approved probe text, no readings) before
    the skip (DELIVERED -> SKIPPED, only at the probe threshold) —
    a skipped question is NEVER a negative answer (no readings, the
    criteria stay OPEN), the re-ask budgets are applied (REVISED ->
    DELIVERED re-ask, then skip as unresolved);
  * the TECHNICAL-FAILURE policy: a technical (model) failure is never
    a participant action — no state change, no budget consumed, the
    model_failure audit event is logged, the participant is asked to
    resend; after the configured number of consecutive failures the
    participant CHOOSES continue (the counter resets) or pause the run
    (RunStatus.PAUSED — frozen, resumable via resume_run(), the
    pending question is re-asked);
  * the steering SETTINGS LOADER (study_content/steering_settings.yaml
    -> SteeringConfig: missing file -> defaults, subset -> merged,
    unknown key / malformed value -> loud error).

All tests are deterministic and need no network access.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from contracts import (
    AgentName,
    AnswerClarity,
    CriterionStatus,
    InputIntent,
    ModelFailureEvent,
    MonitorResult,
    NavigationAssessment,
    QuestionCandidateSet,
    QuestionState,
    RunStatus,
    StateTransitionEvent,
    SteeringConfig,
)
from interview_definition import (
    CriterionDefinition,
    EqRule,
    InterviewDefinition,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ValueDeclaration,
)
from interview_navigator import (
    NavigatorError,
    ReaskKind,
    SkipReason,
    SteeringAction,
    is_explicit_skip,
    normalize_command,
    skip_request_decision,
    steering_decision,
)
from interview_supervisor import InterviewSupervisor
from steering_settings import load_steering_settings

DEFINITION_VERSION = "steer_guide@v1"


def make_steering_definition() -> InterviewDefinition:
    """
    A minimal v2 definition WITHOUT jumps: three required questions in
    the group qg_1 (sq_a, then sq_b, then sq_c) + one criterion on the
    (criteria) group that depends on the value v_mood — the "a skip is
    never a negative answer" assertions read off this criterion.
    """
    return InterviewDefinition(
        definition_version=DEFINITION_VERSION,
        guide_id="steer_guide",
        title="Steering Test Guide",
        values=[ValueDeclaration(name="v_mood", type="bool")],
        sections=[
            SectionGroup(
                id="sec_1",
                title="Mood",
                children=[
                    QuestionGroup(
                        id="qg_1",
                        title="Core",
                        strategy="all_required",
                        criteria_group=True,
                        questions=[
                            QuestionDefinition(
                                id="sq_a", text="Ist Ihre Stimmung gesenkt?"
                            ),
                            QuestionDefinition(
                                id="sq_b", text="Schlafen Sie schlecht?"
                            ),
                            QuestionDefinition(
                                id="sq_c", text="Ist Ihr Appetit gestört?"
                            ),
                        ],
                        criteria=[
                            CriterionDefinition(
                                id="c_mood",
                                rule=EqRule(value="v_mood", equals=True),
                            ),
                        ],
                    ),
                ],
            ),
        ],
        canonical_fallback_message="canonical fallback",
    )


# ---------------------------------------------------------------------------
# the test doubles (the pluggable agent components)
# ---------------------------------------------------------------------------


class ScriptedMonitor:
    """
    The monitor test double: it returns the scripted INTENTS in order
    (the last one repeats) as MonitorResults without signals — the
    run/turn ids are filled in at check time, so one instance works
    for the whole run.
    """

    def __init__(self, intents) -> None:
        assert intents
        self._intents = list(intents)
        self._index = 0
        self.calls = 0

    def check(self, run_id: str, turn_id: str, transcript: str) -> MonitorResult:
        self.calls += 1
        intent = self._intents[min(self._index, len(self._intents) - 1)]
        self._index += 1
        return MonitorResult(signals=(), intent=intent)


class ScriptedDiagnostiker:
    """
    The Diagnostiker test double: it returns the scripted CLARITIES in
    order (the last one repeats) and, optionally, RAISES (a simulated
    technical model failure) on the first `failures` calls.
    """

    def __init__(
        self,
        clarities=(),
        failures: int = 0,
        values: dict | None = None,
    ) -> None:
        self._clarities = list(clarities) or [AnswerClarity.CLEAR]
        self._failures = failures
        self._values = values or {}
        self.calls = 0

    def assess(
        self,
        run_id: str,
        turn_id: str,
        question_id: str,
        transcript: str,
        candidates: QuestionCandidateSet,
    ) -> NavigationAssessment:
        self.calls += 1
        if self.calls <= self._failures:
            raise RuntimeError(
                "simulated technical failure (the model timed out)"
            )
        clarity = self._clarities[
            min(self.calls - self._failures - 1, len(self._clarities) - 1)
        ]
        return NavigationAssessment(
            run_id=run_id,
            turn_id=turn_id,
            answer_clarity=clarity,
            navigation_values=dict(self._values),
            criterion_updates=[],
            next_question_id=None,
            selection_reason=(
                f"test diagnostiker ({clarity.value}) for {question_id}"
            ),
        )


class TestCommandChecks(unittest.TestCase):
    """
    The deterministic command normalization and the explicit-skip
    pre-check (WInf-inspired — deliberately strict: a normal answer
    that contains a command word is never treated as the command).
    """

    def test_normalization_strips_politeness_and_punctuation(self):
        self.assertEqual(
            normalize_command("Bitte überspringen, danke!"), "überspringen danke"
        )
        self.assertEqual(normalize_command("PLEASE   SKIP."), "skip")
        self.assertEqual(normalize_command("  weiter  "), "weiter")
        self.assertEqual(normalize_command(""), "")

    def test_explicit_skip_is_a_normalized_exact_match(self):
        phrases = ("überspringen", "überspringen sie", "skip")
        self.assertTrue(is_explicit_skip("Bitte überspringen", phrases))
        self.assertTrue(is_explicit_skip("Überspringen Sie!", phrases))
        self.assertTrue(is_explicit_skip("please skip.", phrases))
        # a normal answer that CONTAINS the word is not the command:
        self.assertFalse(
            is_explicit_skip("Ich kann die Frage nicht überspringen", phrases)
        )
        self.assertFalse(is_explicit_skip("Ja, überspringen Sie mich", phrases))
        self.assertFalse(is_explicit_skip("   ", phrases))

    def test_failure_choice_phrases_match_after_normalization(self):
        self.assertEqual(normalize_command("Bitte weiter!"), "weiter")
        self.assertEqual(normalize_command("Bitte pausieren"), "pausieren")
        self.assertEqual(normalize_command("Try again!"), "try again")


class TestSteeringDecisionTable(unittest.TestCase):
    """
    The pure steering reaction table: every (intent, clarity, counter
    state) pair of the WInf-inspired policy.
    """

    def setUp(self) -> None:
        self.config = SteeringConfig()  # the built-in defaults

    def test_refusal_is_probed_before_the_skip(self):
        # refusal is a SKIP REQUEST: the question is PROBED first (the
        # re-ask kind skip_probe — ask why + alternatives), the skip
        # happens only once the probe threshold (max_skip_probes=1)
        # is reached:
        probed = steering_decision(
            InputIntent.REFUSAL,
            AnswerClarity.UNCLEAR,
            {"clarify": 1, "redirect": 1},
            self.config,
        )
        self.assertIs(probed.action, SteeringAction.REASK)
        self.assertIs(probed.reask_kind, ReaskKind.SKIP_PROBE)
        # the probe budget is exhausted -> SKIP (refused) — regardless
        # of clarity or the other counters:
        skipped = steering_decision(
            InputIntent.REFUSAL,
            AnswerClarity.UNCLEAR,
            {"skip_probe": 1},
            self.config,
        )
        self.assertIs(skipped.action, SteeringAction.SKIP)
        self.assertIs(skipped.skip_reason, SkipReason.REFUSED)
        # the probe budget is INDEPENDENT of the other kinds:
        independent = steering_decision(
            InputIntent.REFUSAL,
            AnswerClarity.CLEAR,
            {"clarify": 1, "redirect": 1, "gentle": 1, "rephrase": 9},
            self.config,
        )
        self.assertIs(independent.action, SteeringAction.REASK)
        # max_skip_probes=0 = immediate skip (no probe):
        no_probe = SteeringConfig(max_skip_probes=0)
        direct = steering_decision(
            InputIntent.REFUSAL, AnswerClarity.CLEAR, {}, no_probe
        )
        self.assertIs(direct.action, SteeringAction.SKIP)
        self.assertIs(direct.skip_reason, SkipReason.REFUSED)

    def test_skip_request_decision_is_shared_by_the_explicit_command(self):
        # the explicit skip command (not an intent) applies the SAME
        # probe-then-skip decision, with its own skip reason:
        probed = skip_request_decision(
            {}, self.config, SkipReason.EXPLICIT
        )
        self.assertIs(probed.action, SteeringAction.REASK)
        self.assertIs(probed.reask_kind, ReaskKind.SKIP_PROBE)
        skipped = skip_request_decision(
            {"skip_probe": 1}, self.config, SkipReason.EXPLICIT
        )
        self.assertIs(skipped.action, SteeringAction.SKIP)
        self.assertIs(skipped.skip_reason, SkipReason.EXPLICIT)

    def test_clarification_request_rephrases_without_budget(self):
        # WInf RT01: the rephrase re-ask does NOT consume the
        # clarification budget — even with the other counters maxed:
        decision = steering_decision(
            InputIntent.CLARIFICATION_REQUEST,
            AnswerClarity.CLEAR,
            {"clarify": 5, "redirect": 5, "gentle": 5},
            self.config,
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.REPRISE)

    def test_rephrase_is_unlimited_by_default_but_configurable(self):
        unlimited = steering_decision(
            InputIntent.CLARIFICATION_REQUEST,
            AnswerClarity.CLEAR,
            {"rephrase": 99},
            self.config,
        )
        self.assertIs(unlimited.action, SteeringAction.REASK)
        # a FINITE configured threshold still gates it:
        bounded = SteeringConfig(max_rephrasings=2)
        skipped = steering_decision(
            InputIntent.CLARIFICATION_REQUEST,
            AnswerClarity.CLEAR,
            {"rephrase": 2},
            bounded,
        )
        self.assertIs(skipped.action, SteeringAction.SKIP)
        self.assertIs(skipped.skip_reason, SkipReason.UNRESOLVED)

    def test_off_topic_redirect_then_unresolved_skip(self):
        decision = steering_decision(
            InputIntent.OFF_TOPIC, AnswerClarity.CLEAR, {}, self.config
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.REDIRECT)
        # the budget (max_redirects=1) is exhausted -> SKIP (unresolved)
        exhausted = steering_decision(
            InputIntent.OFF_TOPIC,
            AnswerClarity.CLEAR,
            {"redirect": 1},
            self.config,
        )
        self.assertIs(exhausted.action, SteeringAction.SKIP)
        self.assertIs(exhausted.skip_reason, SkipReason.UNRESOLVED)

    def test_unclear_intent_behaves_like_off_topic(self):
        decision = steering_decision(
            InputIntent.UNCLEAR, AnswerClarity.CLEAR, {}, self.config
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.REDIRECT)
        exhausted = steering_decision(
            InputIntent.UNCLEAR,
            AnswerClarity.CLEAR,
            {"redirect": 1},
            self.config,
        )
        self.assertIs(exhausted.action, SteeringAction.SKIP)

    def test_cannot_answer_gentle_then_unresolved_skip(self):
        decision = steering_decision(
            InputIntent.CANNOT_ANSWER, AnswerClarity.CLEAR, {}, self.config
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.GENTLE)
        exhausted = steering_decision(
            InputIntent.CANNOT_ANSWER,
            AnswerClarity.CLEAR,
            {"gentle": 1},
            self.config,
        )
        self.assertIs(exhausted.action, SteeringAction.SKIP)
        self.assertIs(exhausted.skip_reason, SkipReason.UNRESOLVED)

    def test_answer_clear_proceeds(self):
        decision = steering_decision(
            InputIntent.ANSWER, AnswerClarity.CLEAR, {}, self.config
        )
        self.assertIs(decision.action, SteeringAction.PROCEED)
        self.assertIsNone(decision.reask_kind)
        self.assertIsNone(decision.skip_reason)

    def test_answer_unclear_clarify_then_unresolved_skip(self):
        decision = steering_decision(
            InputIntent.ANSWER, AnswerClarity.UNCLEAR, {}, self.config
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.CLARIFICATION)
        exhausted = steering_decision(
            InputIntent.ANSWER,
            AnswerClarity.UNCLEAR,
            {"clarify": 1},
            self.config,
        )
        self.assertIs(exhausted.action, SteeringAction.SKIP)
        self.assertIs(exhausted.skip_reason, SkipReason.UNRESOLVED)

    def test_per_kind_counters_are_independent(self):
        # the redirect budget is exhausted, but the gentle budget
        # (a different kind) still allows a re-ask:
        decision = steering_decision(
            InputIntent.CANNOT_ANSWER,
            AnswerClarity.CLEAR,
            {"redirect": 3, "gentle": 0},
            self.config,
        )
        self.assertIs(decision.action, SteeringAction.REASK)
        self.assertIs(decision.reask_kind, ReaskKind.GENTLE)

    def test_custom_thresholds_are_config_driven(self):
        # max_redirects=2: the second off-topic still re-asks ...
        config = SteeringConfig(max_redirects=2)
        self.assertIs(
            steering_decision(
                InputIntent.OFF_TOPIC,
                AnswerClarity.CLEAR,
                {"redirect": 1},
                config,
            ).action,
            SteeringAction.REASK,
        )
        # ... the third is skipped as unresolved
        self.assertIs(
            steering_decision(
                InputIntent.OFF_TOPIC,
                AnswerClarity.CLEAR,
                {"redirect": 2},
                config,
            ).action,
            SteeringAction.SKIP,
        )


class TestSteeringSettingsLoader(unittest.TestCase):
    """
    The steering settings loader (study_content/steering_settings.yaml
    -> SteeringConfig): missing file -> defaults, a subset -> merged
    over the defaults, an unknown key / a malformed value -> a loud
    error at load time.
    """

    def test_missing_file_returns_the_built_in_defaults(self):
        config = load_steering_settings("/nonexistent/steering.yaml")
        self.assertEqual(config, SteeringConfig())

    def test_a_subset_is_merged_over_the_defaults(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("max_redirects: 2\npause_after_model_failures: 5\n")
            path = f.name
        try:
            config = load_steering_settings(path)
            self.assertEqual(config.max_redirects, 2)
            self.assertEqual(config.pause_after_model_failures, 5)
            # the untouched keys keep the defaults:
            self.assertEqual(config.max_clarifications, 1)
            self.assertEqual(config.max_gentle_followups, 1)
        finally:
            os.unlink(path)

    def test_an_unknown_key_fails_loudly(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("max_reasks: 3\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_steering_settings(path)
        finally:
            os.unlink(path)

    def test_a_malformed_value_fails_loudly(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("max_clarifications: \"many\"\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_steering_settings(path)
        finally:
            os.unlink(path)

    def test_the_study_content_file_loads(self):
        # the actual researcher-editable file ships with the repo:
        config = load_steering_settings()
        self.assertIn("überspringen", config.skip_phrases)
        self.assertEqual(config.pause_after_model_failures, 3)


class TestSteeringPipeline(unittest.TestCase):
    """
    The Supervisor pipeline with the WInf-inspired steering: the
    scripted monitor intents + the scripted diagnostiker clarities
    drive the deterministic reaction table end to end (question
    states, the re-ask budgets, the audit causes, the "never a
    negative answer" invariant).
    """

    def make_supervisor(self, monitor=None, diagnostiker=None):
        return InterviewSupervisor(
            make_steering_definition(),
            monitor=monitor,
            diagnostiker=diagnostiker,
        )

    def transition_causes(self, sup, subject_id):
        """The state-transition causes of one question (the audit "why")."""
        return [
            e.cause
            for e in sup.events
            if isinstance(e, StateTransitionEvent)
            and e.subject_id == subject_id
        ]

    def test_explicit_skip_is_probed_before_the_skip(self):
        """
        (WInf-inspired) the explicit skip command is a SKIP REQUEST:
        the system ASKS the participant why they want to skip (and
        offers the alternative — a vague/short answer is fine), and
        only skips the CURRENT question once the skip-probe threshold
        is reached. Both reactions run BEFORE the Diagnostiker is
        called at all (no LLM).
        """
        diagnostiker = ScriptedDiagnostiker()
        sup = self.make_supervisor(diagnostiker=diagnostiker)
        sup.start()
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        # the FIRST skip request: the question is PROBED — re-asked
        # (FOLLOWUP) with the pre-approved probe text, still DELIVERED
        # (the participant can still answer it):
        out1 = sup.receive_input("Bitte überspringen")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertEqual(out1.question_id, "sq_a")
        # the question text is re-delivered WITH the probe (ask why +
        # the alternative — a vague/short answer is fine):
        self.assertIn("Ist Ihre Stimmung gesenkt?", out1.delivered.text)
        self.assertIn(sup.steering.skip_probe_text, out1.delivered.text)
        # the probe re-ask is audited (the kind + the trigger) ...
        self.assertTrue(
            any(
                "re-ask kind=skip_probe" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        # ... no readings were recorded (the skip request is not an
        # answer — a skip is never a negative answer), and the
        # Diagnostiker was NOT called yet:
        self.assertIsNone(sup.run.value("v_mood"))
        self.assertEqual(diagnostiker.calls, 0)
        # the SECOND skip request: the probe threshold (max_skip_probes=1)
        # is reached — the question is skipped (the explicit skip
        # reason in the audit cause) and the interview advances:
        out2 = sup.receive_input("Bitte überspringen")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertIs(sup.run.question_state("sq_b"), QuestionState.DELIVERED)
        self.assertEqual(out2.question_id, "sq_b")
        self.assertTrue(
            any(
                "skipped (explicit)" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        # ... and the Diagnostiker was NEVER called (the skip-request
        # pre-checks are deterministic, no LLM involved):
        self.assertEqual(diagnostiker.calls, 0)

    def test_the_skip_probe_offers_the_alternative_to_answer(self):
        """
        (WInf-inspired) the probe is NOT a dead end: after the system
        asked why the participant wants to skip (and offered the
        alternative — a vague/short answer is fine), the participant
        can still ANSWER the question: it closes normally.
        """
        diagnostiker = ScriptedDiagnostiker(values={"v_mood": True})
        sup = self.make_supervisor(diagnostiker=diagnostiker)
        sup.start()
        # the skip request -> the probe:
        sup.receive_input("Bitte überspringen")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        # ... and now the participant answers instead (the alternative):
        out = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(out.question_id, "sq_b")
        self.assertIs(
            sup.run.criterion_state("c_mood"), CriterionStatus.SATISFIED
        )

    def test_refusal_is_probed_then_skipped(self):
        """
        (WInf-inspired) the REFUSAL intent (measured by the monitor)
        is a SKIP REQUEST like the explicit skip command: the current
        question is probed first (ask why + alternatives) and skipped
        only once the probe threshold is reached — again before the
        Diagnostiker.
        """
        diagnostiker = ScriptedDiagnostiker()
        monitor = ScriptedMonitor([InputIntent.REFUSAL, InputIntent.REFUSAL])
        sup = self.make_supervisor(monitor=monitor, diagnostiker=diagnostiker)
        sup.start()
        # the FIRST refusal: the probe (re-asked with the pre-approved
        # text, still DELIVERED, no readings):
        out1 = sup.receive_input("Das will ich nicht beantworten.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertIn(sup.steering.skip_probe_text, out1.delivered.text)
        self.assertIsNone(sup.run.value("v_mood"))
        self.assertEqual(diagnostiker.calls, 0)
        # the SECOND refusal: the threshold (max_skip_probes=1) is
        # reached — the question is skipped (refused) and the
        # interview advances:
        out2 = sup.receive_input("Nein, ich will nicht.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertIs(sup.run.question_state("sq_b"), QuestionState.DELIVERED)
        self.assertEqual(out2.question_id, "sq_b")
        self.assertTrue(
            any(
                "skipped (refused)" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        self.assertEqual(diagnostiker.calls, 0)

    def test_off_topic_redirect_then_skip(self):
        """
        (WInf-inspired) an off-topic answer gets ONE redirect re-ask
        (REVISED -> DELIVERED, FOLLOWUP); the second off-topic answer
        exhausts the budget: the question is skipped as UNRESOLVED.
        """
        monitor = ScriptedMonitor([InputIntent.OFF_TOPIC])
        sup = self.make_supervisor(monitor=monitor)
        sup.start()
        out1 = sup.receive_input("Mein Kaktus blüht.")
        # first off-topic: the question is re-asked (the re-ask kind is
        # audited on the revise transition):
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertEqual(out1.question_id, "sq_a")
        self.assertTrue(
            any(
                "re-ask kind=redirect" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        out2 = sup.receive_input("Und mein Kaktus ist grün.")
        # second off-topic: the budget is exhausted -> skip (unresolved)
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertIs(sup.run.question_state("sq_b"), QuestionState.DELIVERED)
        self.assertEqual(out2.question_id, "sq_b")
        self.assertTrue(
            any(
                "skipped (unresolved)" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )

    def test_cannot_answer_gentle_then_skip(self):
        """(WInf-inspired) "cannot answer" gets ONE gentle follow-up."""
        monitor = ScriptedMonitor([InputIntent.CANNOT_ANSWER])
        sup = self.make_supervisor(monitor=monitor)
        sup.start()
        out1 = sup.receive_input("Das weiß ich leider nicht.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertTrue(
            any(
                "re-ask kind=gentle" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        out2 = sup.receive_input("Ich kann es wirklich nicht sagen.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertEqual(out2.question_id, "sq_b")

    def test_clarification_request_rephrases_unlimited(self):
        """
        (WInf-inspired, WInf RT01) the participant asking for a
        rephrasing gets the question re-asked (unlimited by default) —
        the clarification budget is NOT consumed.
        """
        monitor = ScriptedMonitor(
            [
                InputIntent.CLARIFICATION_REQUEST,
                InputIntent.CLARIFICATION_REQUEST,
                InputIntent.CLARIFICATION_REQUEST,
                InputIntent.ANSWER,
            ]
        )
        sup = self.make_supervisor(monitor=monitor)
        sup.start()
        for _ in range(3):
            out = sup.receive_input("Wie soll ich das verstehen?")
            # the question stays in flight (the re-ask), no skip:
            self.assertIs(
                sup.run.question_state("sq_a"), QuestionState.DELIVERED
            )
            self.assertEqual(out.question_id, "sq_a")
        # then a clear answer closes the question normally:
        out = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(out.question_id, "sq_b")

    def test_unclear_answer_clarify_then_skip(self):
        """
        The classic path (intent ANSWER, the Diagnostiker's UNCLEAR
        clarity) runs through the steering table: one clarification
        re-ask, then the question is skipped as unresolved.
        """
        diagnostiker = ScriptedDiagnostiker(
            clarities=[
                AnswerClarity.UNCLEAR,
                AnswerClarity.UNCLEAR,
                AnswerClarity.CLEAR,
            ]
        )
        sup = self.make_supervisor(diagnostiker=diagnostiker)
        sup.start()
        out1 = sup.receive_input("Ich weiß es nicht genau.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertTrue(
            any(
                "re-ask kind=clarify" in cause
                for cause in self.transition_causes(sup, "sq_a")
            )
        )
        out2 = sup.receive_input("Hmm, so etwas vielleicht.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertEqual(out2.question_id, "sq_b")

    def test_per_kind_counters_are_independent_in_the_pipeline(self):
        """
        (WInf-inspired) the per-kind counters are independent: a
        redirect and a gentle follow-up on the SAME question do not
        consume each other's budget.
        """
        monitor = ScriptedMonitor(
            [
                InputIntent.OFF_TOPIC,
                InputIntent.CANNOT_ANSWER,
                InputIntent.OFF_TOPIC,
            ]
        )
        sup = self.make_supervisor(monitor=monitor)
        sup.start()
        out1 = sup.receive_input("Mein Kaktus blüht.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        # a different kind on the same question: the gentle budget is
        # still fresh -> re-ask (not skip):
        out2 = sup.receive_input("Das weiß ich leider nicht.")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertEqual(out2.question_id, "sq_a")
        # the redirect budget IS exhausted now -> skip:
        out3 = sup.receive_input("Aber mein Kaktus ist grün!")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.SKIPPED)
        self.assertEqual(out3.question_id, "sq_b")

    def test_skipped_question_is_never_a_negative_answer(self):
        """
        (WInf-inspired) the core skip invariant: a skipped question
        records NO readings — the criterion that depends on its value
        stays OPEN (never NOT_SATISFIED, never SATISFIED), exactly as
        if the question had not been answered at all.
        """
        sup = self.make_supervisor()
        sup.start()
        sup.receive_input("Bitte überspringen")  # the skip probe (ask why)
        sup.receive_input("Bitte überspringen")  # the threshold: the skip
        # the dependent criterion is OPEN — not a negative answer:
        self.assertIs(sup.run.criterion_state("c_mood"), CriterionStatus.OPEN)
        # the rest of the interview completes normally, WITHOUT the
        # value ever being read — the criterion stays OPEN:
        sup.receive_input("Ja, ich schlafe schlecht.")
        sup.receive_input("Ja, mein Appetit ist gestört.")
        self.assertIs(sup.status, RunStatus.COMPLETED)
        self.assertIs(sup.run.criterion_state("c_mood"), CriterionStatus.OPEN)
        # and NO reading of v_mood was recorded for the skipped
        # question:
        self.assertIsNone(sup.run.value("v_mood"))

    def test_an_answered_question_can_still_satisfy_the_criterion(self):
        """
        The counterpart of the skip invariant: with a NORMAL answer
        (v_mood=True) the dependent criterion is decided SATISFIED —
        so the OPEN state of the skip test is meaningful.
        """
        diagnostiker = ScriptedDiagnostiker(values={"v_mood": True})
        sup = self.make_supervisor(diagnostiker=diagnostiker)
        sup.start()
        sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertIs(
            sup.run.criterion_state("c_mood"), CriterionStatus.SATISFIED
        )


class TestTechnicalFailurePolicy(unittest.TestCase):
    """
    (WInf-inspired) the technical-failure policy in the pipeline: a
    technical (model) failure is NEVER a participant action — no state
    change, no budget consumed, the model_failure audit event is
    logged, the participant is asked to resend; after the configured
    number of CONSECUTIVE failures the participant chooses continue
    (the counter resets) or pause the run (RunStatus.PAUSED — frozen,
    resumable via resume_run(), the pending question is re-asked).
    """

    def make_supervisor(self, diagnostiker):
        return InterviewSupervisor(
            make_steering_definition(), diagnostiker=diagnostiker
        )

    def model_failure_events(self, sup):
        return [e for e in sup.events if isinstance(e, ModelFailureEvent)]

    def test_a_failure_changes_nothing_and_asks_to_resend(self):
        """
        One technical failure: the question stays DELIVERED (still
        awaiting the answer), nothing is recorded, the outcome carries
        the failure facts, the model_failure event is in the audit
        log, and the RESEND prompt is delivered. The next successful
        input processes normally and restarts the failure counter.
        """
        diagnostiker = ScriptedDiagnostiker(failures=1)
        sup = self.make_supervisor(diagnostiker)
        sup.start()
        out = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        # the failure reaction:
        self.assertTrue(out.model_failure)
        self.assertEqual(out.consecutive_model_failures, 1)
        self.assertFalse(out.failure_choice_pending)
        self.assertEqual(out.delivered.text, sup.steering.technical_failure_text)
        # NO state change: the question is still awaiting the answer:
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        self.assertIs(sup.status, RunStatus.ACTIVE)
        # the failure is audited (agent + the failure message):
        failures = self.model_failure_events(sup)
        self.assertEqual(len(failures), 1)
        self.assertIs(failures[0].agent, AgentName.DIAGNOSTIKER)
        self.assertIn("timed out", failures[0].detail)
        # the retry succeeds — the question closes and the counter
        # restarts:
        out2 = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertFalse(out2.model_failure)
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(sup.consecutive_model_failures, 0)
        self.assertEqual(out2.question_id, "sq_b")

    def test_continue_after_three_consecutive_failures(self):
        """
        Choosing "continue" at the threshold: the failure counter
        resets and the pending answer is simply retried (no state
        change — the question stays DELIVERED).
        """
        diagnostiker = ScriptedDiagnostiker(failures=3)
        sup = self.make_supervisor(diagnostiker)
        sup.start()
        for i in range(3):
            out = sup.receive_input(f"Antwort {i + 1}")
        self.assertTrue(out.failure_choice_pending)
        # the participant chooses CONTINUE:
        out4 = sup.receive_input("Bitte weiter!")
        self.assertFalse(out4.failure_choice_pending)
        self.assertIs(sup.status, RunStatus.ACTIVE)
        self.assertEqual(
            out4.delivered.text, sup.steering.continue_confirmation_text
        )
        self.assertEqual(sup.consecutive_model_failures, 0)
        # the retry (the 4th agent call succeeds) processes normally:
        out5 = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertFalse(out5.model_failure)
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(out5.question_id, "sq_b")

    def test_a_non_command_input_repeats_the_choice(self):
        """
        While the failure choice is pending, an input that is neither
        the continue nor the pause command just re-delivers the choice
        (deterministic, no LLM, no state change).
        """
        diagnostiker = ScriptedDiagnostiker(failures=99)
        sup = self.make_supervisor(diagnostiker)
        sup.start()
        for _ in range(3):
            out = sup.receive_input("x")
        self.assertTrue(out.failure_choice_pending)
        out = sup.receive_input("Das ist aber ärgerlich.")
        self.assertTrue(out.failure_choice_pending)
        self.assertIs(sup.status, RunStatus.ACTIVE)
        self.assertEqual(out.delivered.text, sup.steering.failure_choice_text)

    def test_pause_after_three_consecutive_failures(self):
        """
        After `pause_after_model_failures` (default: 3) consecutive
        technical failures the participant CHOOSES. Choosing "pause":
        the run is PAUSED (the transition is logged, the snapshot is
        saved), the pending question stays DELIVERED, and the pipeline
        refuses further input until resume_run().
        """
        diagnostiker = ScriptedDiagnostiker(failures=99)
        sup = self.make_supervisor(diagnostiker)
        sup.start()
        out1 = sup.receive_input("Antwort 1")
        self.assertTrue(out1.model_failure)
        self.assertEqual(out1.consecutive_model_failures, 1)
        out2 = sup.receive_input("Antwort 2")
        self.assertTrue(out2.model_failure)
        self.assertEqual(out2.consecutive_model_failures, 2)
        out3 = sup.receive_input("Antwort 3")
        # the threshold (3) is reached: the choice is offered ...
        self.assertTrue(out3.model_failure)
        self.assertTrue(out3.failure_choice_pending)
        self.assertEqual(out3.delivered.text, sup.steering.failure_choice_text)
        # ... and the participant chooses PAUSE:
        out4 = sup.receive_input("pause")
        self.assertFalse(out4.model_failure)
        self.assertFalse(out4.failure_choice_pending)
        self.assertIs(sup.status, RunStatus.PAUSED)
        self.assertEqual(
            out4.delivered.text, sup.steering.pause_confirmation_text
        )
        # the transition is audited (ACTIVE -> PAUSED) and the pending
        # question is untouched (still awaiting the answer):
        run_transitions = [
            e
            for e in sup.events
            if isinstance(e, StateTransitionEvent)
            and e.subject_id is None
        ]
        self.assertIn(
            ("ACTIVE", "PAUSED"),
            [(e.from_state, e.to_state) for e in run_transitions],
        )
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        # the paused run is frozen for the pipeline:
        with self.assertRaises(NavigatorError):
            sup.receive_input("und jetzt?")
        # resume: PAUSED -> ACTIVE, the pending question is re-asked
        # (a pure output-layer retransmission — no question-state
        # transition, sq_a stays DELIVERED):
        resume = sup.resume_run()
        self.assertIs(resume.run_status, RunStatus.ACTIVE)
        self.assertEqual(resume.question_id, "sq_a")
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.DELIVERED)
        # and the failure counter restarted:
        self.assertEqual(sup.consecutive_model_failures, 0)
        # the interview can continue after the resume:
        sup._diagnostiker = ScriptedDiagnostiker()  # the model recovers
        out5 = sup.receive_input("Ja, meine Stimmung ist gesenkt.")
        self.assertFalse(out5.model_failure)
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(out5.question_id, "sq_b")

    def test_resume_requires_a_paused_run(self):
        """resume_run() fails loudly on a non-PAUSED run."""
        sup = self.make_supervisor(ScriptedDiagnostiker())
        sup.start()
        with self.assertRaises(NavigatorError):
            sup.resume_run()

    def test_a_successful_turn_restarts_the_failure_counter(self):
        """
        The counter counts CONSECUTIVE failures: a successful answer
        in between restarts it — the failure on the question sq_b
        (the 3rd agent call, the 2nd failure overall) is counted as
        the FIRST consecutive failure (1, not 2), so the threshold is
        not reached.
        """

        class AlternatingFailureDiagnostiker(ScriptedDiagnostiker):
            """Fails on every ODD assess call (1st, 3rd, ...), succeeds on
            the even ones (its own counter — the base class's `calls`
            counts the successful calls only)."""

            def __init__(self):
                super().__init__()
                self._assess_calls = 0

            def assess(self, *args, **kwargs):
                self._assess_calls += 1
                if self._assess_calls % 2 == 1:
                    raise RuntimeError("simulated technical failure")
                return super().assess(*args, **kwargs)

        sup = self.make_supervisor(AlternatingFailureDiagnostiker())
        sup.start()
        out1 = sup.receive_input("Antwort 1")
        self.assertTrue(out1.model_failure)
        self.assertEqual(out1.consecutive_model_failures, 1)
        # the retry succeeds (the counter restarts at 0) and the
        # interview advances to sq_b:
        out2 = sup.receive_input("Antwort 2")
        self.assertFalse(out2.model_failure)
        self.assertIs(sup.run.question_state("sq_a"), QuestionState.CLOSED)
        self.assertEqual(sup.consecutive_model_failures, 0)
        # the NEXT failure (sq_b) starts a fresh consecutive run:
        out3 = sup.receive_input("Antwort 3")
        self.assertTrue(out3.model_failure)
        self.assertEqual(out3.consecutive_model_failures, 1)
        self.assertFalse(out3.failure_choice_pending)
