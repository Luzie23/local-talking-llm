"""End-to-end flow tests through the actual Flask application.

The participant talks to /api/interview/start and /api/chat exactly as the
browser does. Only the language model is replaced by a scripted fake that
returns what a competent model is expected to return for each message, so
these tests verify the application's control flow (state, branching,
budget, skips, records, safety, failure handling). They do NOT measure the
quality of a real model and are not clinical validation.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from langchain_core.runnables import RunnableLambda

import app as app_module
import llm_provider
from agents.interview_orchestrator import handle_turn
from interview_guide import INTERVIEWS_DIR, InterviewGuide
from session_record import SessionRecorder

Q1 = "Wie würden Sie Ihre Stimmung in den letzten zwei Wochen beschreiben?"
Q2 = "Gab es Zeiten, in denen Sie sich über nichts mehr freuen konnten?"
Q3 = "Wie war Ihr Schlaf in den letzten zwei Wochen?"

# Expected classification for each scripted participant message:
# (intent, sufficiency, missing_information, evidence_quote, provisional_rating)
SCRIPT = {
    "Sehr schlecht, ich bin fast jeden Tag niedergeschlagen.": ("answer", "sufficient", [], "fast jeden Tag niedergeschlagen", None),
    "Nein.": ("answer", "sufficient", [], "Nein", None),
    "Nein, ich schlafe gut, keine Probleme.": ("answer", "sufficient", [], "ich schlafe gut", None),
    "Gut.": ("answer", "sufficient", [], "Gut", None),
    "Manchmal komisch.": ("answer", "ambiguous", ["what 'komisch' means"], "komisch", None),
    "Ich meine eher müde und gereizt.": ("answer", "sufficient", [], "müde und gereizt", None),
    "Ich verstehe die Frage nicht.": ("clarification_request", "insufficient", ["meaning of the question"], "", None),
    "Weiß ich nicht.": ("cannot_answer", "insufficient", ["mood description"], "", None),
    "Dazu möchte ich nichts sagen.": ("refusal", "insufficient", [], "", None),
    "Ich habe Hotdogs gegessen.": ("off_topic", "insufficient", ["mood description"], "", None),
    "Hm.": ("unclear", "insufficient", ["mood description"], "", None),
    "Hotdogs machen mich nicht glücklich.": ("answer", "ambiguous", ["whether pleasure is reduced in general, not only for one food"], "Hotdogs", None),
    "Ich denke manchmal an Suizid.": ("answer", "sufficient", [], "Ich denke manchmal an Suizid", None),
    "Ich will nicht darüber reden, ich denke an Suizid.": ("refusal", "insufficient", [], "", None),
    # synthetic SCID-format test items
    "Ja, ich meide große Gruppen fast immer, zum Beispiel bei der Arbeit und im Verein.": ("answer", "sufficient", [], "ich meide große Gruppen fast immer", "2"),
    "Ja, ich bin sehr selbstbewusst.": ("answer", "sufficient", [], "I am never anxious", "0"),
    "Ich verstehe die Frage immer noch nicht.": ("clarification_request", "insufficient", ["meaning of the question"], "", None,),
    "Heute habe ich Pizza gegessen.": ("off_topic", "insufficient", ["mood description"], "", None, ),
    "Ich habe oft Konflikte mit meiner Familie, und das belastet mich.": ( "answer", "ambiguous", ["what happens during family conflicts"], "Konflikte mit meiner Familie", None,),
}

QUESTION_AGENT_REPLIES = {
    "did not understand": "Gemeint ist, wie Sie sich gefühlsmäßig meistens gefühlt haben: Wie ging es Ihnen in den letzten zwei Wochen?",
    "missing detail": "Was meinen Sie genau mit diesem Gefühl?",
    "cannot answer": "Wenn Sie nur an die letzten Tage denken, wie war Ihre Stimmung da?",
    "unrelated": "Lassen Sie uns zur Frage zurückkehren: Wie haben Sie sich in den letzten zwei Wochen gefühlt?",
}


class ScriptedModel:
    """Fake model: routes each prompt to the scripted reply of its agent."""

    def __init__(self):
        """Start in normal mode and keep every prompt each agent received, so tests can inspect them."""
        self.mode = "ok"  # "ok" | "garbage" | "raise"
        self.judgment_prompts = []
        self.question_prompts = []
        self.ack_prompts = []

    def __call__(self, prompt):
        """Route the prompt to the scripted reply of the agent it belongs to (Judgment / Question / acknowledgement)."""
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        if "You are the Judgment Agent" in text:
            self.judgment_prompts.append(text)
            if self.mode == "garbage":
                return "Sure! The participant seems fine."
            if self.mode == "raise":
                raise ConnectionError("model server not reachable")
            latest = re.findall(r"<<<(.*?)>>>", text, re.S)[-1]
            intent, suff, missing, quote, rating = SCRIPT[latest]
            return json.dumps({"intent": intent, "sufficiency": suff, "missing_information": missing,
                               "evidence_quote": quote, "provisional_rating": rating,
                               "reasoning": "scripted"})
        if "You are the Question Agent" in text:
            self.question_prompts.append(text)
            for key, reply in QUESTION_AGENT_REPLIES.items():
                if key in text:
                    return reply
            return "Können Sie das bitte genauer beschreiben?"
        self.ack_prompts.append(text)
        return "FEEDBACK: Danke für Ihre Antwort.\nFOLLOWUP: NONE"


class FlowTestBase(unittest.TestCase):
    """Starts a fresh interview on the 3-question development guide through the real Flask app, with a scripted model and a temporary record folder."""
    def setUp(self):
        """Prepare a fresh interview, scripted model and temporary record location for each test."""
        self.model = ScriptedModel()
        fake = RunnableLambda(self.model)
        app_module.llm = fake
        app_module.chain_with_history = llm_provider.build_chain(fake)
        self.records_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.records_dir, ignore_errors=True)
        # These flows use the 3-question development guide regardless of the
        # guide that is active in study_settings.yaml.
        app_module.interview = InterviewGuide("depression_scid_example.yaml")
        app_module.recorder = SessionRecorder(app_module.interview, directory=self.records_dir)
        self.client = app_module.app.test_client()
        opening = self.client.get("/api/interview/start").get_json()["message"]
        self.assertIn(Q1, opening)

    def say(self, text):
        """Send one participant message to /api/chat and return the JSON reply."""
        res = self.client.post("/api/chat", json={"text": text})
        self.assertEqual(res.status_code, 200)
        return res.get_json()

    def current(self):
        """Id of the item that is currently being asked (None at the end)."""
        q = app_module.interview.current_question()
        return q["id"] if q else None

    def records(self):
        """The item records collected so far."""
        return app_module.interview.records

    def saved_record(self):
        """Load the single session-record JSON file written during the test."""
        files = os.listdir(self.records_dir)
        self.assertEqual(len(files), 1)
        with open(os.path.join(self.records_dir, files[0]), encoding="utf-8") as f:
            return json.load(f)


class TestClearAnswers(FlowTestBase):
    """Clear positive, clear negative and short-but-sufficient answers."""
    def test_positive_negative_and_short_answers_complete_the_interview(self):
        """Positive negative and short answers complete the interview."""
        r = self.say("Sehr schlecht, ich bin fast jeden Tag niedergeschlagen.")
        self.assertIn(Q2, r["response"])
        r = self.say("Nein.")  # clear negative: no probing for a symptom
        self.assertIn(Q3, r["response"])
        r = self.say("Gut.")   # short but sufficient
        self.assertTrue(r["finished"])
        self.assertIn("letzte Frage", r["response"])
        self.assertEqual([x["status"] for x in self.records()], ["answered"] * 3)
        self.assertEqual([x["clarifications_used"] for x in self.records()], [0, 0, 0])
        self.assertEqual(self.model.question_prompts, [])
        saved = self.saved_record()
        self.assertTrue(saved["finished"])
        self.assertIn("NOT SCID", saved["content_source"])
        self.assertIn("Not a diagnosis", saved["notice"])

    def test_clear_negative_sleep_answer_is_accepted(self):
        """Clear negative sleep answer is accepted."""
        self.say("Gut.")
        self.say("Nein.")
        r = self.say("Nein, ich schlafe gut, keine Probleme.")
        self.assertTrue(r["finished"])
        self.assertEqual(self.records()[-1]["status"], "answered")
        self.assertEqual(self.records()[-1]["judgment"]["sufficiency"], "sufficient")


class TestClarifications(FlowTestBase):
    """Ambiguous answers, explanation requests, inability to answer, off-topic replies, narrow examples and budget exhaustion."""
    def test_ambiguous_answer_gets_one_focused_clarification(self):
        """Ambiguous answer gets one focused clarification."""
        r = self.say("Manchmal komisch.")
        self.assertEqual(self.current(), "dep_01")
        self.assertEqual(r["response"], QUESTION_AGENT_REPLIES["missing detail"])
        self.assertIn("what 'komisch' means", self.model.question_prompts[0])
        r = self.say("Ich meine eher müde und gereizt.")
        self.assertIn(Q2, r["response"])
        # The clarification answer was judged together with the first answer.
        self.assertIn("<<<Manchmal komisch.>>>", self.model.judgment_prompts[-1])
        rec = self.records()[0]
        self.assertEqual((rec["status"], rec["clarifications_used"], len(rec["turns"])), ("answered", 1, 2))

    def test_second_ambiguous_answer_skips_current_item(self):
        """Second unresolved ambiguous answer skips the current item."""
        self.say("Manchmal komisch.")
        r = self.say("Manchmal komisch.")

        self.assertEqual(self.current(), "dep_02")
        self.assertIn(Q2, r["response"])
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records()[0]["status"], "skipped")
        self.assertEqual(self.records()[0]["skip_reason"], "unresolved")

    # EMSAG: RT01: Repeated explanation requests remain on the same item.
    def test_repeated_explanation_requests_keep_rephrasing(self):
        """Repeated explanation requests do not force the interview to skip."""
        for _ in range(3):
            r = self.say("Ich verstehe die Frage nicht.")
            self.assertEqual(self.current(), "dep_01")
            self.assertNotIn(Q1, r["response"])
            self.assertEqual(
                r["response"],
                QUESTION_AGENT_REPLIES["did not understand"],
            )

        self.assertEqual(self.records(), [])
        self.assertEqual(app_module.interview.current_clarification_count(), 3)

    def test_cannot_answer_gets_one_gentle_followup_then_skips(self):
        """First cannot-answer response gets one gentle follow-up; second skips."""
        first = self.say("Weiß ich nicht.")

        self.assertEqual(self.current(), "dep_01")
        self.assertEqual(
            first["response"],
            QUESTION_AGENT_REPLIES["cannot answer"],
        )
        self.assertEqual(self.records(), [])
        self.assertEqual(app_module.interview.current_interaction_count("gentle"), 1)

        second = self.say("Weiß ich nicht.")

        self.assertEqual(self.current(), "dep_02")
        self.assertIn(Q2, second["response"])
        self.assertEqual(self.records()[0]["status"], "skipped")
        self.assertEqual(self.records()[0]["skip_reason"], "cannot_answer")

    def test_second_off_topic_response_skips_current_item(self):
        """Second consecutive off-topic response skips the current item."""
        first = self.say("Ich habe Hotdogs gegessen.")
        self.assertEqual(self.current(), "dep_01")
        self.assertEqual(
            first["response"],
            QUESTION_AGENT_REPLIES["unrelated"],
        )

        second = self.say("Heute habe ich Pizza gegessen.")
        self.assertEqual(self.current(), "dep_02")
        self.assertIn(Q2, second["response"])
        self.assertEqual(self.records()[0]["status"], "skipped")
        self.assertEqual(self.records()[0]["skip_reason"], "unresolved")

    def test_narrow_example_is_not_treated_as_general_loss_of_pleasure(self):
        """Narrow example is not treated as general loss of pleasure."""
        self.say("Gut.")
        self.say("Hotdogs machen mich nicht glücklich.")
        self.assertEqual(self.current(), "dep_02")  # not advanced on a narrow example
        self.assertIn("not only for one food", self.model.question_prompts[-1])


class TestRefusalAndSkip(FlowTestBase):
    """Refusals and explicit skip commands."""
    def test_refusal_skips_immediately_without_persuasion(self):
        """Refusal skips immediately without persuasion."""
        r = self.say("Dazu möchte ich nichts sagen.")
        self.assertIn("völlig in Ordnung", r["response"])
        self.assertIn(Q2, r["response"])
        self.assertEqual(self.model.question_prompts, [])
        rec = self.records()[0]
        self.assertEqual((rec["status"], rec["skip_reason"]), ("skipped", "refused"))

    def test_explicit_skip_command_needs_no_model(self):
        """Explicit skip command needs no model."""
        self.model.mode = "raise"  # even with the model down
        r = self.say("Bitte überspringen.")
        self.assertIn(Q2, r["response"])
        self.assertEqual(self.model.judgment_prompts, [])
        self.assertEqual(self.records()[0]["skip_reason"], "refused")

    def test_skip_during_pending_clarification(self):
        """Skip during pending clarification."""
        self.say("Manchmal komisch.")
        self.say("überspringen")
        self.assertEqual(self.current(), "dep_02")
        self.assertEqual(self.records()[0]["skip_reason"], "refused")


class TestSafety(FlowTestBase):
    """The keyword safety check stays active alongside the agent workflow."""
    def test_safety_flag_is_raised_and_interview_continues(self):
        """Safety flag is raised and interview continues."""
        r = self.say("Ich denke manchmal an Suizid.")
        self.assertTrue(r["safety_flag"])
        self.assertIn(Q2, r["response"])

    def test_refusal_does_not_disable_safety_check(self):
        """Refusal does not disable safety check."""
        r = self.say("Ich will nicht darüber reden, ich denke an Suizid.")
        self.assertTrue(r["safety_flag"])
        self.assertEqual(self.records()[0]["skip_reason"], "refused")


class TestModelFailures(FlowTestBase):
    """Malformed or failed model responses must never advance the interview."""
    def test_malformed_output_does_not_advance_or_use_budget_and_pauses_after_three(self):
        """Malformed output does not advance or use budget and pauses after three."""
        self.model.mode = "garbage"
        for _ in range(2):
            r = self.say("Gut.")
            self.assertIn("technisches Problem", r["response"])
            self.assertIn(Q1, r["response"])
            self.assertEqual(self.current(), "dep_01")
            self.assertEqual(app_module.interview.current_clarification_count(), 0)
        r = self.say("Gut.")
        self.assertIn("pausiert", r["response"])
        self.assertEqual(self.records(), [])  # nothing recorded as a participant result
        events = self.saved_record()["system_events"]
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e["kind"] == "judgment_failure" for e in events))
        # Recovery: model back, interview still paused until restarted.
        self.model.mode = "ok"
        self.assertIn("pausiert", self.say("Gut.")["response"])

    def test_temporary_failure_then_success_continues_normally(self):
        """Temporary failure then success continues normally."""
        self.model.mode = "raise"
        self.say("Gut.")
        self.model.mode = "ok"
        r = self.say("Gut.")
        self.assertIn(Q2, r["response"])
        rec = self.records()[0]
        self.assertEqual(len(rec["turns"]), 1)  # the failed attempt is not stored as evidence


class TestProvisionalRatings(unittest.TestCase):
    """Rating path with SYNTHETIC test items in SCID-5-PD format (not SCID wording)."""

    GUIDE = """
guide_id: synthetic_pd
language: de
rating_scheme: scid5pd
sections:
  - questions:
      - {id: T1, text: "Synthetische Testfrage eins?", criterion: "synthetic avoidance", rating_anchor: "2 = several examples"}
      - {id: T2, text: "Synthetische Testfrage zwei?", criterion: "synthetic"}
      - {id: T3, text: "Synthetische Testfrage drei?", criterion: "synthetic"}
"""

    def setUp(self):
        """Prepare a fresh interview, scripted model and temporary record location for each test."""
        fd, path = tempfile.mkstemp(suffix=".yaml", dir=INTERVIEWS_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self.GUIDE)
        self.addCleanup(os.remove, path)
        self.guide = InterviewGuide(os.path.basename(path))
        self.recorder = SessionRecorder(self.guide, enabled=False)
        self.llm = RunnableLambda(ScriptedModel())

    def test_ratings_are_provisional_and_skips_are_unknown(self):
        """Ratings are provisional and skips are unknown."""
        handle_turn(self.llm, self.guide, self.recorder,
                    "Ja, ich meide große Gruppen fast immer, zum Beispiel bei der Arbeit und im Verein.")
        handle_turn(self.llm, self.guide, self.recorder, "Ja, ich bin sehr selbstbewusst.")
        handle_turn(self.llm, self.guide, self.recorder, "überspringen")
        ratings = [r["provisional_rating"] for r in self.guide.records]
        self.assertEqual(ratings[0]["value"], "2")
        self.assertIn("provisional", ratings[0]["label"])
        # invented evidence ("I am never anxious") -> rating forced to "?"
        self.assertEqual(ratings[1]["value"], "?")
        self.assertEqual(ratings[2]["value"], "?")  # skipped is never "0"
        self.assertNotIn("total", json.dumps(self.recorder.to_dict()).lower())

# EMSAG: Regression test to ensure that starting a speech interview
    # creates only one session. The audio startup endpoint must not reset
    # the interview or create a second session record. It must reuse the
    # existing interview state and synthesize the opening question.
class TestSessionStartup(unittest.TestCase):
    """Session must start only once when the speech interview is opened."""

    def setUp(self):
        self.client = app_module.app.test_client()
        self.records_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.records_dir, ignore_errors=True)

        app_module.interview = InterviewGuide("depression_scid_example.yaml")
        app_module.recorder = SessionRecorder(
            app_module.interview,
            directory=self.records_dir
        )

    def test_audio_start_does_not_start_a_second_session(self):
        start_calls = []
        original_start = app_module._start_new_session

        def tracked_start(*args, **kwargs):
            start_calls.append(True)
            return original_start(*args, **kwargs)

        app_module._start_new_session = tracked_start
        self.addCleanup(
            setattr,
            app_module,
            "_start_new_session",
            original_start
        )

        start_response = self.client.get("/api/interview/start")
        self.assertEqual(start_response.status_code, 200)
        self.assertEqual(len(start_calls), 0)

        session_response = self.client.post(
            "/api/interview/session",
            json={
                "historical_retrieval_consent": False
            }
        )

        self.assertEqual(session_response.status_code, 200)
        self.assertEqual(len(start_calls), 1)

        first_question = app_module.interview.current_question()
        self.assertIsNotNone(first_question)

        class FakeTTS:
            def synthesize(self, text):
                return 22050, [0.0] * 100

        original_get_tts = app_module.get_tts
        original_encode = app_module.speech.encode_wav_base64

        app_module.get_tts = lambda: FakeTTS()
        app_module.speech.encode_wav_base64 = lambda sr, audio: "fake-audio"

        self.addCleanup(setattr, app_module, "get_tts", original_get_tts)
        self.addCleanup(
            setattr,
            app_module.speech,
            "encode_wav_base64",
            original_encode
        )

        audio_response = self.client.get("/api/interview/start/audio")
        self.assertEqual(audio_response.status_code, 200)

        pieces = audio_response.get_json()["pieces"]

        self.assertEqual(len(start_calls), 1)
        self.assertEqual(
            app_module.interview.current_question()["id"],
            first_question["id"]
        )
        self.assertTrue(any(p["kind"] == "question" for p in pieces))
        self.assertTrue(all(p["audio"] == "fake-audio" for p in pieces))
            # EMSAG: Verify session startup count directly; JSON files are created only when save() is called.
            #self.assertEqual(len(os.listdir(self.records_dir)), 1)
            # EMSAG: Verify session startup count directly; JSON files are created only when save() is called.
# EMSAG: Regression test to ensure that starting a speech interview
    # creates only one session. The audio startup endpoint must not reset
    # the interview or create a second session record. It must reuse the
    # existing interview state and synthesize the opening question.

if __name__ == "__main__":
    unittest.main()
