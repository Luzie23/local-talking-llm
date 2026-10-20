"""
Unit tests for backend/contracts.py (RECONSTRUCTION_PLAN.md Phase 1,
steps 1.1 + 1.3): the boundary types, the ID scheme, the three separate
intervention signals, and the audit event schema.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pydantic import TypeAdapter, ValidationError

import contracts
from contracts import (
    AbortRequestSignal,
    AgentName,
    AnswerClarity,
    ApprovedContentPackage,
    ContentKind,
    ContentPiece,
    CriterionStatus,
    DeliveryState,
    EventType,
    InterventionSignalType,
    PromptInjectionSignal,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionState,
    RunState,
    RunStatus,
    SpeakerMode,
    SpeakerOutput,
    StudyCondition,
    SuicidalitySignal,
    TransparencyMode,
    ValidationAspect,
    ValidationDecision,
    ValidationOutcome,
    delivery_id,
    history_key,
    new_run_id,
    parse_event,
    turn_id,
)

# The typed union the three separate signals are exposed as.
SIGNAL_ADAPTER = TypeAdapter(contracts.InterventionSignal)


def _event_base(**overrides) -> dict:
    """Valid identity fields for building events in the tests."""
    base = {
        "run_id": new_run_id(),
        "turn_id": turn_id(1),
        "definition_version": "test_guide@v1",
    }
    base.update(overrides)
    return base


class TestStateVocabularies(unittest.TestCase):
    """The state enums must carry exactly the values the plan names."""

    def test_criterion_status_is_four_valued(self):
        self.assertEqual(
            {s.value for s in CriterionStatus},
            {"SATISFIED", "NOT_SATISFIED", "OPEN", "UNCLEAR"},
        )

    def test_open_and_unclear_are_distinct_values(self):
        # OPEN (not asked yet) vs UNCLEAR (answered but uninterpretable)
        # are separate states — that distinction is architectural (rule 1).
        self.assertNotEqual(CriterionStatus.OPEN, CriterionStatus.UNCLEAR)

    def test_question_states_include_revised(self):
        # REVISED: already asked, information still lacking - the question
        # is asked again before the navigator may move on (see its docstring).
        self.assertEqual(
            {s.value for s in QuestionState},
            {
                "UNSEEN",
                "REQUIRED",
                "RESERVED",
                "DELIVERED",
                "ANSWER_RECORDED",
                "CLOSED",
                "REVISED",
                "SKIPPED",
            },
        )

    def test_run_and_delivery_states(self):
        self.assertEqual(
            {s.value for s in RunStatus}, {"ACTIVE", "COMPLETED", "ABORTED"}
        )
        self.assertEqual(
            {s.value for s in DeliveryState},
            {"RESERVED", "VALIDATED", "DELIVERED", "ACKNOWLEDGED"},
        )

    def test_modalities_are_text_and_voice_only(self):
        # Terminal is development-only - not a study modality.
        self.assertEqual(
            {m.value for m in contracts.Modality}, {"text", "voice"}
        )

    def test_transparency_modes_are_the_two_disclosure_setups(self):
        self.assertEqual(
            {t.value for t in TransparencyMode},
            {"FULL_DISCLOSURE", "BASIC_DISCLOSURE"},
        )

    def test_validation_outcomes(self):
        self.assertEqual(
            {o.value for o in ValidationOutcome}, {"ALLOW", "REVISE", "BLOCK"}
        )

    def test_event_types_include_the_input_events(self):
        # 14 lifecycle events + input_received + input_transcribed.
        self.assertEqual(len(EventType), 16)
        self.assertIn("input_received", {e.value for e in EventType})
        self.assertIn("input_transcribed", {e.value for e in EventType})


class TestIds(unittest.TestCase):
    """The ID scheme from day one (step 1.3)."""

    def test_run_id_format(self):
        for _ in range(20):
            self.assertRegex(new_run_id(), contracts.RUN_ID_RE.pattern)

    def test_run_ids_are_unique(self):
        self.assertEqual(len({new_run_id() for _ in range(2000)}), 2000)

    def test_turn_id_format(self):
        self.assertEqual(turn_id(1), "turn_001")
        self.assertEqual(turn_id(999), "turn_999")
        self.assertEqual(turn_id(1000), "turn_1000")
        with self.assertRaises(ValueError):
            turn_id(0)

    def test_delivery_id_format(self):
        rid = new_run_id()
        did = delivery_id(rid, 1)
        self.assertEqual(did, f"{rid}__d001")
        self.assertRegex(did, contracts.DELIVERY_ID_RE.pattern)

    def test_delivery_id_rejects_bad_run_id(self):
        with self.assertRaises(ValueError):
            delivery_id("not-a-run-id", 1)
        with self.assertRaises(ValueError):
            delivery_id(new_run_id(), 0)

    def test_history_key_is_run_scoped(self):
        rid = new_run_id()
        self.assertEqual(
            history_key(rid, AgentName.DIAGNOSTIKER), f"{rid}::diagnostiker"
        )
        # Per-agent AND per-run: the legacy fixed session id is gone.
        self.assertNotIn("voice_assistant_session", history_key(rid, AgentName.SPRECHER))
        self.assertNotEqual(
            history_key(rid, AgentName.SPRECHER), history_key(rid, AgentName.VALIDATOR)
        )

    def test_models_reject_malformed_ids(self):
        with self.assertRaises(ValidationError) as ctx:
            RunState(run_id="not-a-run-id", definition_version="v1")
        self.assertIn("run_id", str(ctx.exception))


class TestInterventionSignals(unittest.TestCase):
    """The three SEPARATE typed signals — never one merged safety score."""

    def _identity(self) -> dict:
        return {"run_id": new_run_id(), "turn_id": turn_id(1)}

    def test_exactly_three_signal_types(self):
        self.assertEqual(
            {t.value for t in InterventionSignalType},
            {"SUICIDALITY", "ABORT_REQUEST", "PROMPT_INJECTION"},
        )

    def test_each_signal_constructs_and_roundtrips(self):
        cases = [
            (SuicidalitySignal, {"matched_keywords": ["stimmung schlecht"]}),
            (AbortRequestSignal, {"detected_phrase": "ich möchte aufhören"}),
            (PromptInjectionSignal, {"indicators": ["ignore all previous instructions"]}),
        ]
        for cls, extra in cases:
            signal = cls(**self._identity(), **extra)
            data = signal.model_dump(mode="json")
            parsed = SIGNAL_ADAPTER.validate_python(data)
            self.assertIsInstance(parsed, cls)
            self.assertEqual(parsed.model_dump(mode="json"), data)

    def test_union_rejects_unknown_signal_type_with_message(self):
        bad = {**self._identity(), "signal_type": "MERGED_SAFETY_SCORE"}
        with self.assertRaises(ValidationError) as ctx:
            SIGNAL_ADAPTER.validate_python(bad)
        msg = str(ctx.exception)
        for allowed in ("SUICIDALITY", "ABORT_REQUEST", "PROMPT_INJECTION"):
            self.assertIn(allowed, msg)

    def test_no_merged_safety_score_field_exists(self):
        # The merged-score concept is structurally absent: extra fields are
        # forbidden, so e.g. a 'safety_score' cannot be smuggled in.
        with self.assertRaises(ValidationError):
            SuicidalitySignal(**self._identity(), safety_score=0.9)

    def test_signals_carry_their_own_fields(self):
        suicidality = SuicidalitySignal(**self._identity())
        injection = PromptInjectionSignal(**self._identity())
        self.assertEqual(suicidality.matched_keywords, [])
        self.assertEqual(injection.indicators, [])
        self.assertNotEqual(type(suicidality), type(injection))


class TestNavigationAssessment(unittest.TestCase):
    """The Diagnostiker's structured output (Phase 3.2 boundary)."""

    def _identity(self) -> dict:
        return {"run_id": new_run_id(), "turn_id": turn_id(2)}

    def test_minimal_assessment(self):
        assessment = contracts.NavigationAssessment(
            **self._identity(),
            answer_clarity=AnswerClarity.CLEAR,
            selection_reason="next required question of the current group",
        )
        self.assertIsNone(assessment.next_question_id)
        self.assertEqual(assessment.navigation_values, {})
        self.assertEqual(assessment.criterion_updates, [])

    def test_full_assessment_with_values_and_updates(self):
        from contracts import CriterionUpdate

        assessment = contracts.NavigationAssessment(
            **self._identity(),
            answer_clarity=AnswerClarity.UNCLEAR,
            navigation_values={"sleep_score": 3, "mentioned_social withdrawal": False},
            criterion_updates=[
                CriterionUpdate(
                    criterion_id="c_sleep_disturbance",
                    status=CriterionStatus.UNCLEAR,
                    evidence="mal weiß ich nicht",
                    reasoning="answer exists but is not interpretable",
                )
            ],
            next_question_id="sleep_01",
            selection_reason="clarify the unclear sleep answer",
        )
        data = assessment.model_dump(mode="json")
        self.assertEqual(
            contracts.NavigationAssessment.model_validate(data).model_dump(mode="json"),
            data,
        )

    def test_selection_reason_is_required(self):
        with self.assertRaises(ValidationError) as ctx:
            contracts.NavigationAssessment(
                **self._identity(), answer_clarity=AnswerClarity.CLEAR
            )
        self.assertIn("selection_reason", str(ctx.exception))

    def test_invalid_clarity_rejected(self):
        with self.assertRaises(ValidationError):
            contracts.NavigationAssessment(
                **self._identity(),
                answer_clarity="VERY_CLEAR",
                selection_reason="x",
            )


class TestCandidateSet(unittest.TestCase):
    """The Navigator's candidate-set types (Phase 2.4 boundary)."""

    def test_set_with_candidates(self):
        run = new_run_id()
        set_ = QuestionCandidateSet(
            run_id=run,
            turn_id=turn_id(1),
            candidates=[
                QuestionCandidate(
                    question_id="dep_01",
                    text="Wie würden Sie Ihre Stimmung beschreiben?",
                    reason_required="required by group 'stimmung'",
                    ordering=0,
                ),
                QuestionCandidate(
                    question_id="dep_02",
                    text="Gab es Zeiten der Anhedonie?",
                    group_id="stimmung",
                    reason_required="all_required of group 'stimmung'",
                    ordering=1,
                ),
            ],
        )
        self.assertFalse(set_.is_empty())
        self.assertEqual(set_.get("dep_02").group_id, "stimmung")
        self.assertIsNone(set_.get("dep_99"))

    def test_empty_set(self):
        set_ = QuestionCandidateSet(run_id=new_run_id(), turn_id=turn_id(1))
        self.assertTrue(set_.is_empty())


class TestSpeakerAndValidator(unittest.TestCase):
    """ApprovedContentPackage / SpeakerOutput / ValidationDecision."""

    def _identity(self) -> dict:
        return {"run_id": new_run_id(), "turn_id": turn_id(1)}

    def test_package_defaults_to_all_invariants(self):
        package = ApprovedContentPackage(
            **self._identity(),
            content=[
                ContentPiece(
                    kind=ContentKind.QUESTION,
                    text="Wie war Ihr Schlaf in den letzten zwei Wochen?",
                    source=contracts.ContentSource.GUIDE_VERBATIM,
                )
            ],
        )
        self.assertEqual(set(package.invariants), set(contracts.InvariantAspect))
        self.assertEqual(package.target_language, "de")

    def test_package_requires_content(self):
        with self.assertRaises(ValidationError):
            ApprovedContentPackage(**self._identity(), content=[])

    def test_speaker_output_defaults_to_reformulated(self):
        out = SpeakerOutput(
            **self._identity(),
            content=[ContentPiece(kind=ContentKind.FEEDBACK, text="Danke für Ihre Antwort.")],
        )
        self.assertIs(out.mode, SpeakerMode.REFORMULATED)

    def test_speaker_output_requires_content(self):
        with self.assertRaises(ValidationError):
            SpeakerOutput(**self._identity(), content=[])

    def test_allow_decision_without_reasons(self):
        decision = ValidationDecision(**self._identity(), outcome=ValidationOutcome.ALLOW)
        self.assertEqual(len(decision.checked_aspects), 5)
        self.assertEqual(decision.reasons, [])

    def test_revise_requires_instructions(self):
        with self.assertRaises(ValidationError) as ctx:
            ValidationDecision(**self._identity(), outcome=ValidationOutcome.REVISE)
        self.assertIn("revision_instructions", str(ctx.exception))
        ok = ValidationDecision(
            **self._identity(),
            outcome=ValidationOutcome.REVISE,
            revision_instructions="preserve the negation in the question",
        )
        self.assertEqual(ok.revision_instructions, "preserve the negation in the question")

    def test_block_requires_reasons(self):
        from contracts import ValidationReason

        with self.assertRaises(ValidationError) as ctx:
            ValidationDecision(**self._identity(), outcome=ValidationOutcome.BLOCK)
        self.assertIn("reason", str(ctx.exception))
        blocked = ValidationDecision(
            **self._identity(),
            outcome=ValidationOutcome.BLOCK,
            reasons=[ValidationReason(code="MEANING_DRIFT", detail="temporal reference changed")],
            canonical_fallback_used=True,
        )
        self.assertTrue(blocked.canonical_fallback_used)

    def test_fallback_flag_only_with_block(self):
        with self.assertRaises(ValidationError):
            ValidationDecision(
                **self._identity(),
                outcome=ValidationOutcome.ALLOW,
                canonical_fallback_used=True,
            )

    def test_decision_roundtrips(self):
        from contracts import ValidationReason

        decision = ValidationDecision(
            **self._identity(),
            outcome=ValidationOutcome.BLOCK,
            reasons=[ValidationReason(code="NON_DIAGNOSIS", detail="diagnostic wording")],
            checked_aspects=[ValidationAspect.NON_DIAGNOSIS],
            canonical_fallback_used=True,
        )
        data = decision.model_dump(mode="json")
        self.assertEqual(
            ValidationDecision.model_validate(data).model_dump(mode="json"), data
        )


class TestStudyCondition(unittest.TestCase):
    def test_speaker_on_by_default(self):
        condition = StudyCondition(transparency=TransparencyMode.FULL_DISCLOSURE)
        self.assertTrue(condition.speaker_enabled)
        self.assertEqual(condition.interview_language, "de")
        self.assertIsNone(condition.behavior_profile)

    def test_transparency_has_no_default(self):
        # FULL_DISCLOSURE vs. BASIC_DISCLOSURE is an explicit study decision.
        with self.assertRaises(ValidationError):
            StudyCondition()

    def test_basic_disclosure_framing(self):
        condition = StudyCondition(
            transparency=TransparencyMode.BASIC_DISCLOSURE, speaker_enabled=False
        )
        self.assertFalse(condition.speaker_enabled)


class TestRunState(unittest.TestCase):
    def test_bumped_advances_version_exactly_once(self):
        state = RunState(run_id=new_run_id(), definition_version="guide@v2")
        self.assertEqual(state.run_version, 1)
        new_state = state.bumped(status=RunStatus.COMPLETED, current_turn_id=None)
        self.assertEqual(new_state.run_version, 2)
        self.assertIs(new_state.status, RunStatus.COMPLETED)
        # the old snapshot is untouched (frozen)
        self.assertIs(state.status, RunStatus.ACTIVE)

    def test_state_is_immutable(self):
        state = RunState(run_id=new_run_id(), definition_version="guide@v2")
        with self.assertRaises(ValidationError):
            state.run_id = new_run_id()  # type: ignore[misc]

    def test_run_version_must_be_positive(self):
        with self.assertRaises(ValidationError):
            RunState(run_id=new_run_id(), definition_version="v", run_version=0)

    def test_state_roundtrips(self):
        state = RunState(
            run_id=new_run_id(),
            definition_version="guide@v2",
            question_states={"dep_01": QuestionState.DELIVERED},
            criterion_states={"c_1": CriterionStatus.OPEN},
            current_turn_id=turn_id(3),
        )
        data = state.model_dump(mode="json")
        self.assertEqual(RunState.model_validate(data).model_dump(mode="json"), data)


class TestEventSchema(unittest.TestCase):
    """One event per transition / agent output / decision / signal / delivery step."""

    def test_every_event_type_roundtrips_through_parse_event(self):
        run = new_run_id()
        t1 = turn_id(1)
        d1 = delivery_id(run, 1)
        events = [
            contracts.RunStartedEvent(
                run_id=run,
                definition_version="g@v1",
                study_condition=StudyCondition(transparency=TransparencyMode.FULL_DISCLOSURE),
            ),
            contracts.RunCompletedEvent(run_id=run, definition_version="g@v1", reason="done"),
            contracts.RunAbortedEvent(
                run_id=run, definition_version="g@v1",
                cause=contracts.RunAbortCause.PARTICIPANT_ABORT_INTERVIEW,
                reason="participant aborted", feedback="not feeling well"),
            contracts.TurnStartedEvent(run_id=run, turn_id=t1, definition_version="g@v1"),
            contracts.TurnCompletedEvent(run_id=run, turn_id=t1, definition_version="g@v1"),
            contracts.InputReceivedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                modality=contracts.Modality.TEXT, content="gut"),
            contracts.InputReceivedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                modality=contracts.Modality.VOICE, audio_ref="recordings/turn_001.wav"),
            contracts.InputTranscribedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                transcript="Meine Stimmung war die meiste Zeit schlecht.", language="de"),
            contracts.StateTransitionEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                subject=contracts.SubjectKind.QUESTION, subject_id="dep_01",
                from_state="DELIVERED", to_state="ANSWER_RECORDED",
                cause="answer received"),
            contracts.AgentOutputEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                agent=AgentName.DIAGNOSTIKER, raw_output='{"ok": true}',
                parsed={"answer_clarity": "CLEAR"}),
            contracts.ValidationDecisionEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                decision=ValidationDecision(run_id=run, turn_id=t1, outcome=ValidationOutcome.ALLOW)),
            contracts.InterventionSignalEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                signal=SuicidalitySignal(run_id=run, turn_id=t1, matched_keywords=["schlecht"])),
            contracts.DeliveryReservedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                delivery_id=d1, content_kind=ContentKind.QUESTION, content_text="Frage?"),
            contracts.DeliveryValidatedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                delivery_id=d1, outcome=ValidationOutcome.ALLOW),
            contracts.DeliveryDeliveredEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                delivery_id=d1, modality=contracts.Modality.VOICE),
            contracts.DeliveryAcknowledgedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1", delivery_id=d1),
            contracts.DeliveryRetransmittedEvent(
                run_id=run, turn_id=t1, definition_version="g@v1",
                delivery_id=d1, attempts=1),
        ]
        for event in events:
            data = event.model_dump(mode="json")
            parsed = parse_event(data)
            self.assertEqual(type(parsed), type(event))
            self.assertEqual(parsed.model_dump(mode="json"), data)

    def test_parse_event_rejects_unknown_type_and_lists_known(self):
        data = _event_base(event_type="does_not_exist")
        with self.assertRaises(ValueError) as ctx:
            parse_event(data)
        msg = str(ctx.exception)
        self.assertIn("run_started", msg)
        self.assertIn("agent_output", msg)
        self.assertIn("delivery_acknowledged", msg)

    def test_agent_output_rejects_parsed_and_error_together(self):
        with self.assertRaises(ValidationError) as ctx:
            contracts.AgentOutputEvent(
                **_event_base(),
                agent=AgentName.SPRECHER,
                parsed={"content": []},
                parse_error="boom",
            )
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_agent_output_with_parse_error_is_fine(self):
        event = contracts.AgentOutputEvent(
            **_event_base(),
            agent=AgentName.VALIDATOR,
            raw_output="not json at all",
            parse_error="expecting value: line 1 column 1",
        )
        self.assertIsNone(event.parsed)

    def test_every_event_requires_definition_version(self):
        with self.assertRaises(ValidationError) as ctx:
            contracts.TurnCompletedEvent(run_id=new_run_id())
        self.assertIn("definition_version", str(ctx.exception))

    def test_event_id_must_be_positive(self):
        with self.assertRaises(ValidationError):
            contracts.TurnCompletedEvent(
                run_id=new_run_id(), definition_version="v", event_id=0
            )

    def test_event_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            contracts.TurnCompletedEvent(
                **_event_base(), safety_score=0.1
            )

    def test_events_are_frozen(self):
        event = contracts.TurnCompletedEvent(
            run_id=new_run_id(), definition_version="v", event_id=1
        )
        with self.assertRaises(ValidationError):
            event.event_id = 2  # type: ignore[misc]


class TestInputAndAbortContracts(unittest.TestCase):
    """Input events (text/voice) and the abort/suicidality protocol."""

    def test_text_input_requires_content(self):
        with self.assertRaises(ValidationError):
            contracts.InputReceivedEvent(
                **_event_base(), modality=contracts.Modality.TEXT, content=""
            )
        with self.assertRaises(ValidationError):
            contracts.InputReceivedEvent(**_event_base(), modality=contracts.Modality.TEXT)

    def test_voice_input_requires_audio_ref(self):
        with self.assertRaises(ValidationError):
            contracts.InputReceivedEvent(
                **_event_base(), modality=contracts.Modality.VOICE
            )
        event = contracts.InputReceivedEvent(
            **_event_base(),
            modality=contracts.Modality.VOICE,
            audio_ref="recordings/turn_001.wav",
        )
        self.assertEqual(event.audio_ref, "recordings/turn_001.wav")

    def test_transcribed_event_requires_transcript(self):
        with self.assertRaises(ValidationError):
            contracts.InputTranscribedEvent(**_event_base(), transcript="")

    def test_turn_started_no_longer_carries_input(self):
        with self.assertRaises(ValidationError):
            contracts.TurnStartedEvent(**_event_base(), participant_input="x")

    def test_suicidality_action_defaults_to_hint_only(self):
        signal = SuicidalitySignal(run_id=new_run_id(), turn_id=turn_id(1))
        self.assertIs(signal.action, contracts.SuicidalityAction.HINT_ONLY)
        early = SuicidalitySignal(
            run_id=new_run_id(), turn_id=turn_id(1),
            action=contracts.SuicidalityAction.EARLY_STOP,
        )
        self.assertIs(early.action, contracts.SuicidalityAction.EARLY_STOP)

    def test_run_aborted_requires_cause(self):
        with self.assertRaises(ValidationError):
            contracts.RunAbortedEvent(
                run_id=new_run_id(), definition_version="v", reason="x"
            )

    def test_run_aborted_causes_and_feedback(self):
        event = contracts.RunAbortedEvent(
            run_id=new_run_id(), definition_version="v",
            cause=contracts.RunAbortCause.PARTICIPANT_ABORT_STUDY,
            reason="participant aborted the study",
        )
        self.assertIsNone(event.feedback)
        self.assertEqual(event.cause.value, "participant_abort_study")
        event2 = contracts.RunAbortedEvent(
            run_id=new_run_id(), definition_version="v",
            cause=contracts.RunAbortCause.PARTICIPANT_ABORT_INTERVIEW,
            reason="participant aborted", feedback="please stop",
        )
        self.assertEqual(event2.feedback, "please stop")
        event3 = contracts.RunAbortedEvent(
            run_id=new_run_id(), definition_version="v",
            cause=contracts.RunAbortCause.SAFETY_EARLY_STOP,
            reason="suicidality: early stop per safety protocol",
        )
        self.assertEqual(event3.cause.value, "safety_early_stop")


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
