"""Unit tests for the LLM provider factory in backend/llm_provider.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the backend/ folder to the path so we can import llm_provider directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestCreateLlm(unittest.TestCase):
    """Tests for the create_llm() factory function."""

    def _import_create_llm(self):
        """
        Import create_llm(). Note: unlike the old app.py, llm_provider.py has
        no dependency on audio/ML libraries (whisper, torch, chatterbox, ...),
        so no heavy mocking is needed here anymore — that's one of the
        benefits of separating the LLM logic from the speech logic.
        """
        if "llm_provider" in sys.modules:
            del sys.modules["llm_provider"]
        from llm_provider import create_llm
        return create_llm

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_default_model(self, mock_ollama):
        """Test that Ollama provider uses default model 'gemma3'."""
        create_llm = self._import_create_llm()
        create_llm("ollama")
        mock_ollama.assert_called_once_with(model="gemma3", base_url="http://localhost:11434")

    @patch("langchain_ollama.OllamaLLM")
    def test_ollama_provider_custom_model(self, mock_ollama):
        """Test that Ollama provider accepts a custom model name."""
        create_llm = self._import_create_llm()
        create_llm("ollama", model="llama3")
        mock_ollama.assert_called_once_with(model="llama3", base_url="http://localhost:11434")

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_provider_default_model(self, mock_chat):
        """Test that MiniMax provider uses default model 'MiniMax-M2.7'."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key")
        mock_chat.assert_called_once_with(
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            api_key="test-key",
            temperature=0.7,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_provider_custom_model(self, mock_chat):
        """Test that MiniMax provider accepts a custom model name."""
        create_llm = self._import_create_llm()
        create_llm("minimax", model="MiniMax-M2.7-highspeed", api_key="test-key")
        mock_chat.assert_called_once_with(
            model="MiniMax-M2.7-highspeed",
            base_url="https://api.minimax.io/v1",
            api_key="test-key",
            temperature=0.7,
        )

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_api_key_from_env(self, mock_chat):
        """Test that MiniMax provider reads API key from MINIMAX_API_KEY env var."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            create_llm("minimax")
        mock_chat.assert_called_once_with(
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            api_key="env-key",
            temperature=0.7,
        )

    def test_minimax_no_api_key_raises(self):
        """Test that MiniMax provider raises ValueError when no API key is provided."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure MINIMAX_API_KEY is not set
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                create_llm("minimax")
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_zero(self, mock_chat):
        """Test that temperature 0.0 is clamped to 0.01 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=0.0)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_negative(self, mock_chat):
        """Test that negative temperature is clamped to 0.01 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=-0.5)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_clamping_high(self, mock_chat):
        """Test that temperature > 1.0 is clamped to 1.0 for MiniMax."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=2.0)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 1.0)

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_temperature_normal(self, mock_chat):
        """Test that a valid temperature is passed through unchanged."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key", temperature=0.5)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.5)

    def test_unknown_provider_raises(self):
        """Test that an unknown provider raises ValueError."""
        create_llm = self._import_create_llm()
        with self.assertRaises(ValueError) as ctx:
            create_llm("unknown_provider")
        self.assertIn("Unknown provider", str(ctx.exception))

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_api_key_arg_overrides_env(self, mock_chat):
        """Test that explicit api_key argument takes precedence over env var."""
        create_llm = self._import_create_llm()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            create_llm("minimax", api_key="arg-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "arg-key")

    @patch("langchain_openai.ChatOpenAI")
    def test_minimax_base_url(self, mock_chat):
        """Test that MiniMax provider uses the correct base URL."""
        create_llm = self._import_create_llm()
        create_llm("minimax", api_key="test-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")


# speech.py needs a microphone/speaker library (sounddevice) which may not be
# installed in a plain test environment, so a stand-in is registered here,
# once, before speech.py is imported. This is done at module load time (not
# inside patch.dict) because numpy's C extension cannot be re-imported
# within the same process, and patch.dict would otherwise undo the import.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()
import speech  # noqa: E402  (must come after the sounddevice stand-in above)


class TestAnalyzeEmotion(unittest.TestCase):
    """Tests for the analyze_emotion() helper function (now in backend/speech.py)."""

    def _import_analyze_emotion(self):
        return speech.analyze_emotion

    def test_neutral_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("The weather is nice today.")
        self.assertAlmostEqual(score, 0.5)

    def test_emotional_text(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("This is amazing! I love it!")
        self.assertGreater(score, 0.5)

    def test_score_capped_at_max(self):
        analyze_emotion = self._import_analyze_emotion()
        # Text with many emotional keywords
        score = analyze_emotion("amazing terrible love hate excited sad happy angry wonderful awful !")
        self.assertLessEqual(score, 0.9)

    def test_score_has_minimum(self):
        analyze_emotion = self._import_analyze_emotion()
        score = analyze_emotion("ok")
        self.assertGreaterEqual(score, 0.3)


class TestParseFeedbackAndFollowup(unittest.TestCase):
    """Tests for llm_provider.parse_feedback_and_followup() (Step 3)."""

    def _import_parse_fn(self):
        if "llm_provider" in sys.modules:
            del sys.modules["llm_provider"]
        from llm_provider import parse_feedback_and_followup
        return parse_feedback_and_followup

    def test_no_followup(self):
        parse = self._import_parse_fn()
        feedback, followup = parse("FEEDBACK: That sounds understandable.\nFOLLOWUP: NONE")
        self.assertEqual(feedback, "That sounds understandable.")
        self.assertIsNone(followup)

    def test_with_followup(self):
        parse = self._import_parse_fn()
        feedback, followup = parse(
            "FEEDBACK: That's clear.\nFOLLOWUP: How long did that last?"
        )
        self.assertEqual(feedback, "That's clear.")
        self.assertEqual(followup, "How long did that last?")

    def test_german_none_marker(self):
        parse = self._import_parse_fn()
        feedback, followup = parse("FEEDBACK: Verstanden.\nFOLLOWUP: Keine")
        self.assertIsNone(followup)

    def test_missing_followup_line(self):
        parse = self._import_parse_fn()
        feedback, followup = parse("FEEDBACK: Danke für die Antwort.")
        self.assertEqual(feedback, "Danke für die Antwort.")
        self.assertIsNone(followup)

    def test_model_ignores_format_entirely(self):
        """If the model doesn't use the format at all, fail safe: treat everything as feedback."""
        parse = self._import_parse_fn()
        feedback, followup = parse("That sounds difficult for you.")
        self.assertEqual(feedback, "That sounds difficult for you.")
        self.assertIsNone(followup)

    def test_strips_ai_prefix_before_parsing(self):
        parse = self._import_parse_fn()
        feedback, followup = parse("AI: FEEDBACK: Thanks for sharing.\nFOLLOWUP: NONE")
        self.assertEqual(feedback, "Thanks for sharing.")
        self.assertIsNone(followup)


class TestLanguageEnforcement(unittest.TestCase):
    """
    Regression tests for the language-drift bug: the model was observed
    replying in English even when the participant used German, despite the
    (implicit) instruction to match the participant's language. Fixed by
    explicitly telling the model which language to use, based on
    study_settings.yaml, instead of relying on it to infer this correctly.
    """

    def _import_get_feedback_and_followup(self):
        if "llm_provider" in sys.modules:
            del sys.modules["llm_provider"]
        import llm_provider
        return llm_provider

    def test_german_language_directive_is_sent(self):
        lp = self._import_get_feedback_and_followup()

        captured = {}

        class FakeChain:
            def invoke(self, inputs, config=None):
                captured.update(inputs)
                return "FEEDBACK: ok\nFOLLOWUP: NONE"

        class FakeProfile:
            instructions = "Some instructions."

        lp.get_feedback_and_followup(
            FakeChain(), FakeProfile(), "Question?", "Answer.",
            followup_allowed=False, language_code="de",
        )
        self.assertIn("German", captured["instructions"])

    def test_english_language_directive_is_sent(self):
        lp = self._import_get_feedback_and_followup()

        captured = {}

        class FakeChain:
            def invoke(self, inputs, config=None):
                captured.update(inputs)
                return "FEEDBACK: ok\nFOLLOWUP: NONE"

        class FakeProfile:
            instructions = "Some instructions."

        lp.get_feedback_and_followup(
            FakeChain(), FakeProfile(), "Question?", "Answer.",
            followup_allowed=False, language_code="en",
        )
        self.assertIn("English", captured["instructions"])

    def test_unknown_language_code_falls_back_to_the_code_itself(self):
        lp = self._import_get_feedback_and_followup()

        captured = {}

        class FakeChain:
            def invoke(self, inputs, config=None):
                captured.update(inputs)
                return "FEEDBACK: ok\nFOLLOWUP: NONE"

        class FakeProfile:
            instructions = "Some instructions."

        lp.get_feedback_and_followup(
            FakeChain(), FakeProfile(), "Question?", "Answer.",
            followup_allowed=False, language_code="fr",
        )
        self.assertIn("fr", captured["instructions"])


if __name__ == "__main__":
    unittest.main()
