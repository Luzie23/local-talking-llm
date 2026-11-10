"""
Unit tests for backend/llm_clients.py (RECONSTRUCTION_PLAN.md Phase 3,
step 3.1): the per-agent provider configuration (incl. the owner-added
"litellm" and "huggingface" OpenAI-compatible providers), the strict
structured JSON generation against the Phase 1 contracts, the
deterministic FakeLLM test double, and the settings file.

All tests are deterministic: the real code path (JSON extraction,
schema validation, bounded retries, fail-closed behavior) runs against
the scripted FakeLLM — no network, no model, no I/O. The OpenAICaller
tests construct the real client object but never hit the network
(skipped when the 'openai' package is not installed, the same pattern
as test_llm_provider.py for langchain_openai).
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# Add the backend/ folder to the path so we can import llm_clients
# directly (the same convention as the other test modules).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pydantic import ValidationError  # noqa: E402

from contracts import NavigationAssessment, new_run_id, turn_id  # noqa: E402
from llm_clients import (  # noqa: E402
    FakeLLM,
    JSONExtractionError,
    LLMCallError,
    LLMConfigurationError,
    LLMSettings,
    LLMProvider,
    OpenAICaller,
    PROVIDER_DEFAULTS,
    ProviderConfig,
    ResolvedProviderConfig,
    StructuredLLMClient,
    StructuredOutputError,
    build_agent_clients,
    create_client,
    extract_json_object,
    load_llm_settings,
    resolve_api_key,
)

_HAS_OPENAI = importlib.util.find_spec("openai") is not None
_SKIP_OPENAI = "openai not installed — skipping (test-only cloud providers)"


def _ollama_config(**overrides) -> ResolvedProviderConfig:
    """A resolved Ollama config with a 0s retry backoff (fast tests)."""
    base = dict(
        provider=LLMProvider.OLLAMA,
        model="llama3.2:3b",
        temperature=0.0,
        timeout_s=300.0,
        max_attempts=3,
        max_output_tokens=1024,
        retry_backoff_s=0.0,
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        json_mode=False,
    )
    base.update(overrides)
    return ResolvedProviderConfig(**base)


def _valid_assessment_dict(run_id: str, turn_id: str) -> dict:
    """A schema-valid NavigationAssessment as a plain dict."""
    return {
        "run_id": run_id,
        "turn_id": turn_id,
        "answer_clarity": "CLEAR",
        "navigation_values": {"v_mood": True},
        "criterion_updates": [
            {
                "criterion_id": "c_anh",
                "status": "SATISFIED",
                "evidence": "ja, seit drei Wochen",
                "reasoning": "affirmative statement about the mood",
            }
        ],
        "next_question_id": "sq_1.1.2",
        "selection_reason": "next open question in the required set",
    }



class TestProviderResolution(unittest.TestCase):
    """The three-layer merge: built-ins < defaults < per-agent entry."""

    def test_defaults_resolve_to_the_local_ollama_model(self):
        settings = LLMSettings()  # the built-in local-first fallback
        config = settings.resolved_for("diagnostiker")
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "llama3.2:3b")
        self.assertEqual(config.base_url, "http://localhost:11434/v1")
        self.assertIsNone(config.api_key_env)
        self.assertEqual(
            config.timeout_s, PROVIDER_DEFAULTS["ollama"]["timeout_s"]
        )
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.max_output_tokens, 1024)
        self.assertEqual(config.temperature, 0.0)
        self.assertFalse(config.json_mode)

    def test_defaults_layer_wins_over_the_provider_builtins(self):
        settings = LLMSettings(
            defaults=ProviderConfig(
                provider=LLMProvider.MINIMAX,
                model="MiniMax-M2.7-highspeed",
                timeout_s=30.0,
            )
        )
        config = settings.resolved_for("sprecher")
        self.assertEqual(config.provider, "minimax")
        self.assertEqual(config.base_url, "https://api.minimax.io/v1")
        self.assertEqual(config.api_key_env, "MINIMAX_API_KEY")
        self.assertEqual(config.model, "MiniMax-M2.7-highspeed")
        self.assertEqual(config.timeout_s, 30.0)  # not the built-in 120s

    def test_agent_entry_overrides_the_defaults(self):
        settings = LLMSettings(
            defaults=ProviderConfig(
                provider=LLMProvider.OLLAMA, model="llama3.2:3b"
            ),
            validator=ProviderConfig(
                provider=LLMProvider.LITELLM, model="glm-5.3"
            ),
        )
        validator = settings.resolved_for("validator")
        self.assertEqual(validator.provider, "litellm")
        self.assertEqual(validator.model, "glm-5.3")
        self.assertEqual(
            validator.base_url, "https://llm-api.ai-lab.uni-luebeck.de"
        )
        self.assertEqual(validator.api_key_env, "LITELLM_API_KEY")
        # the other agents inherit the defaults untouched:
        self.assertEqual(settings.resolved_for("diagnostiker").model, "llama3.2:3b")

    def test_empty_agent_block_inherits_the_defaults(self):
        settings = LLMSettings(
            defaults=ProviderConfig(
                provider=LLMProvider.OLLAMA, model="llama3.2:3b"
            ),
            sprecher=ProviderConfig(),  # {} in the YAML
        )
        config = settings.resolved_for("sprecher")
        self.assertEqual(config.model, "llama3.2:3b")

    def test_huggingface_builtins(self):
        settings = LLMSettings(
            defaults=ProviderConfig(
                provider=LLMProvider.HUGGINGFACE,
                model="Qwen/Qwen3-VL-8B-Instruct",
            )
        )
        config = settings.resolved_for("diagnostiker")
        self.assertEqual(
            config.base_url, "https://api-inference.huggingface.co/v1"
        )
        self.assertEqual(config.api_key_env, "HF_TOKEN")

    def test_unknown_provider_value_is_rejected(self):
        with self.assertRaises(ValidationError):
            ProviderConfig(provider="gpt")
        with self.assertRaises(ValidationError):
            LLMSettings(
                defaults=ProviderConfig(provider=LLMProvider.OLLAMA),
                validator=ProviderConfig(provider="mistral"),
            )

    def test_unknown_settings_field_is_rejected(self):
        # a typo must not silently configure nothing (extra="forbid"):
        with self.assertRaises(ValidationError):
            ProviderConfig(model="llama3.2:3b", modle="llama3.2:3b")
        with self.assertRaises(ValidationError):
            LLMSettings(
                defaults=ProviderConfig(),
                bogus_agent=ProviderConfig(),
            )

    def test_out_of_range_values_are_rejected(self):
        with self.assertRaises(ValidationError):
            ProviderConfig(temperature=3.0)
        with self.assertRaises(ValidationError):
            ProviderConfig(max_attempts=0)
        with self.assertRaises(ValidationError):
            ProviderConfig(timeout_s=-1)

    def test_missing_provider_fails_loudly(self):
        settings = LLMSettings(defaults=ProviderConfig())
        with self.assertRaises(LLMConfigurationError) as ctx:
            settings.resolved_for("diagnostiker")
        self.assertIn("no provider configured", str(ctx.exception))

    def test_litellm_without_a_model_fails_loudly(self):
        # the deployment's model names must be explicit:
        settings = LLMSettings(
            defaults=ProviderConfig(provider=LLMProvider.LITELLM)
        )
        with self.assertRaises(LLMConfigurationError) as ctx:
            settings.resolved_for("sprecher")
        self.assertIn("no model configured", str(ctx.exception))

    def test_unknown_agent_label_fails_loudly(self):
        settings = LLMSettings()
        with self.assertRaises(LLMConfigurationError):
            settings.resolved_for("diagnostikerr")


class TestApiKeyResolution(unittest.TestCase):
    """Keys live ONLY in the environment — never in files or the audit."""

    def test_ollama_needs_no_key(self):
        self.assertIsNone(resolve_api_key(_ollama_config()))

    def test_key_is_read_from_the_named_environment_variable(self):
        config = _ollama_config(
            provider=LLMProvider.MINIMAX,
            model="MiniMax-M2.7",
            api_key_env="MINIMAX_API_KEY",
        )
        with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-test-123"}):
            self.assertEqual(resolve_api_key(config), "sk-test-123")

    def test_missing_key_fails_loudly_at_configuration_time(self):
        config = _ollama_config(
            provider=LLMProvider.MINIMAX, api_key_env="MINIMAX_API_KEY"
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError) as ctx:
                resolve_api_key(config)
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_litellm_uses_its_own_env_var_by_default(self):
        config = _ollama_config(
            provider=LLMProvider.LITELLM,
            model="glm-5.3",
            base_url="https://llm-api.ai-lab.uni-luebeck.de",
            api_key_env="LITELLM_API_KEY",
        )
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": "sk-litellm-456"}):
            self.assertEqual(resolve_api_key(config), "sk-litellm-456")

    def test_custom_api_key_env_name(self):
        config = _ollama_config(
            provider=LLMProvider.HUGGINGFACE,
            model="Qwen/Qwen3-VL-8B-Instruct",
            base_url="https://api-inference.huggingface.co/v1",
            api_key_env="MY_HF_KEY",
        )
        with mock.patch.dict(os.environ, {"MY_HF_KEY": "hf-token"}):
            self.assertEqual(resolve_api_key(config), "hf-token")



class TestJSONExtraction(unittest.TestCase):
    """The raw-reply -> JSON-object extraction (tolerant but strict)."""

    def test_bare_json_object(self):
        raw = json.dumps({"answer_clarity": "CLEAR"})
        self.assertEqual(json.loads(extract_json_object(raw)), {"answer_clarity": "CLEAR"})

    def test_json_fence(self):
        raw = 'Here is the result:\n```json\n{"ok": true}\n```'
        self.assertEqual(json.loads(extract_json_object(raw)), {"ok": True})

    def test_plain_fence_without_language_tag(self):
        raw = "```\n{\"ok\": true}\n```"
        self.assertEqual(json.loads(extract_json_object(raw)), {"ok": True})

    def test_prose_lead_in(self):
        raw = 'Sure! The assessment is: {"ok": true} — done.'
        self.assertEqual(json.loads(extract_json_object(raw)), {"ok": True})

    def test_nested_braces_and_braces_in_strings(self):
        raw = json.dumps(
            {
                "outer": {"inner": {"deeper": True}},
                "text": "the model said {something} like this}",
            }
        )
        self.assertEqual(json.loads(extract_json_object(raw)), json.loads(raw))

    def test_no_braces_at_all(self):
        with self.assertRaises(JSONExtractionError):
            extract_json_object("Danke für Ihre Antwort.")

    def test_unbalanced_object(self):
        with self.assertRaises(JSONExtractionError):
            extract_json_object('{"ok": true')

    def test_object_that_does_not_parse(self):
        with self.assertRaises(JSONExtractionError):
            extract_json_object('{"ok": true,}')

    def test_empty_reply(self):
        with self.assertRaises(JSONExtractionError):
            extract_json_object("")


class TestFakeLLM(unittest.TestCase):
    """The deterministic test double (no I/O, scripted, recorded)."""

    def test_serves_scripted_responses_in_order(self):
        fake = FakeLLM(["first", "second"])
        self.assertEqual(
            fake.complete("s", "u", temperature=0.0, max_tokens=64), "first"
        )
        self.assertEqual(
            fake.complete("s", "u", temperature=0.0, max_tokens=64), "second"
        )
        self.assertEqual(fake.responses_left, 0)

    def test_records_every_call(self):
        fake = FakeLLM(["x"])
        fake.complete("system", "user", temperature=0.2, max_tokens=128)
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call.system_prompt, "system")
        self.assertEqual(call.user_prompt, "user")
        self.assertEqual(call.temperature, 0.2)
        self.assertEqual(call.max_tokens, 128)

    def test_scripted_exception_is_raised(self):
        failure = LLMCallError("simulated timeout")
        fake = FakeLLM([failure, "recovered"])
        with self.assertRaises(LLMCallError) as ctx:
            fake.complete("s", "u", temperature=0.0, max_tokens=64)
        self.assertEqual(str(ctx.exception), "simulated timeout")
        self.assertEqual(
            fake.complete("s", "u", temperature=0.0, max_tokens=64),
            "recovered",
        )

    def test_exhausted_script_fails(self):
        fake = FakeLLM()
        with self.assertRaises(LLMCallError) as ctx:
            fake.complete("s", "u", temperature=0.0, max_tokens=64)
        self.assertIn("exhausted", str(ctx.exception))

    def test_reset_replaces_the_script_and_clears_calls(self):
        fake = FakeLLM(["old"])
        fake.complete("s", "u", temperature=0.0, max_tokens=64)
        fake.reset(["new"])
        self.assertEqual(len(fake.calls), 0)
        self.assertEqual(
            fake.complete("s", "u", temperature=0.0, max_tokens=64), "new"
        )



class TestStructuredGeneration(unittest.TestCase):
    """
    The strict structured output: the real client code path (extraction,
    Pydantic validation against the Phase 1 contract, bounded retries,
    fail-closed) runs against the scripted FakeLLM.
    """

    def _client(self, raw_responses, **config_overrides) -> StructuredLLMClient:
        config = _ollama_config(**config_overrides)
        return StructuredLLMClient(
            "diagnostiker", config, caller=FakeLLM(raw_responses)
        )

    def test_valid_json_parses_to_the_contract(self):
        run_id = new_run_id()
        turn = turn_id(1)
        raw = json.dumps(_valid_assessment_dict(run_id, turn))
        client = self._client([raw])
        result = client.generate(
            "You are the Diagnostiker.",
            "Evaluate this answer.",
            NavigationAssessment,
        )
        self.assertIsInstance(result.parsed, NavigationAssessment)
        self.assertEqual(result.parsed.run_id, run_id)
        self.assertEqual(result.parsed.answer_clarity.value, "CLEAR")
        self.assertEqual(result.parsed.next_question_id, "sq_1.1.2")
        self.assertTrue(result.parsed.navigation_values["v_mood"])
        self.assertEqual(result.raw, raw)  # the audit keeps the raw output
        self.assertEqual(result.attempts, 1)

    def test_fenced_json_parses(self):
        run_id = new_run_id()
        turn = turn_id(1)
        raw = (
            "```json\n"
            + json.dumps(_valid_assessment_dict(run_id, turn))
            + "\n```"
        )
        client = self._client([raw])
        result = client.generate("s", "u", NavigationAssessment)
        self.assertEqual(result.parsed.run_id, run_id)
        self.assertEqual(result.attempts, 1)

    def test_unknown_field_is_rejected_fail_closed(self):
        # the contracts are extra="forbid": a diagnosis field the model
        # must never emit fails loudly (no silent acceptance):
        run_id = new_run_id()
        turn = turn_id(1)
        bad = _valid_assessment_dict(run_id, turn)
        bad["diagnosis"] = "manic episode"
        bad_json = json.dumps(bad)
        # both attempts return the invalid JSON (attempts = bounded):
        client = self._client([bad_json, bad_json], max_attempts=2)
        with self.assertRaises(StructuredOutputError) as ctx:
            client.generate("s", "u", NavigationAssessment)
        error = ctx.exception
        self.assertEqual(error.agent, "diagnostiker")
        self.assertEqual(error.attempts, 2)
        self.assertEqual(error.last_raw, json.dumps(bad))
        self.assertIn("diagnosis", error.error)

    def test_wrong_enum_value_is_rejected(self):
        run_id = new_run_id()
        turn = turn_id(1)
        bad = _valid_assessment_dict(run_id, turn)
        bad["answer_clarity"] = "MAYBE"
        client = self._client([json.dumps(bad)], max_attempts=1)
        with self.assertRaises(StructuredOutputError) as ctx:
            client.generate("s", "u", NavigationAssessment)
        self.assertIn("answer_clarity", ctx.exception.error)


    def test_retry_after_a_non_json_reply(self):
        run_id = new_run_id()
        turn = turn_id(1)
        good = json.dumps(_valid_assessment_dict(run_id, turn))
        client = self._client(["I am thinking...", good], max_attempts=3)
        result = client.generate("s", "u", NavigationAssessment)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(client._caller.calls), 2)
        # the correction note (with the concrete error) was fed back:
        second = client._caller.calls[1].user_prompt
        self.assertTrue(second.startswith("u"))
        self.assertIn(
            "IMPORTANT: your previous reply was unusable", second
        )

    def test_retry_after_a_schema_violation(self):
        run_id = new_run_id()
        turn = turn_id(1)
        bad = _valid_assessment_dict(run_id, turn)
        bad["answer_clarity"] = "MAYBE"
        good = json.dumps(_valid_assessment_dict(run_id, turn))
        client = self._client([json.dumps(bad), good], max_attempts=3)
        result = client.generate("s", "u", NavigationAssessment)
        self.assertEqual(result.attempts, 2)
        self.assertIn(
            "answer_clarity", client._caller.calls[1].user_prompt
        )

    def test_attempts_are_bounded_fail_closed(self):
        client = self._client(["no json", "still no json"], max_attempts=2)
        with self.assertRaises(StructuredOutputError) as ctx:
            client.generate("s", "u", NavigationAssessment)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertEqual(ctx.exception.last_raw, "still no json")
        self.assertEqual(len(client._caller.calls), 2)

    def test_transport_error_is_retried(self):
        run_id = new_run_id()
        turn = turn_id(1)
        good = json.dumps(_valid_assessment_dict(run_id, turn))
        client = self._client(
            [LLMCallError("simulated timeout"), good], max_attempts=3
        )
        result = client.generate("s", "u", NavigationAssessment)
        self.assertEqual(result.attempts, 2)

    def test_transport_error_exhausts_and_fails_closed(self):
        client = self._client(
            [LLMCallError("timeout"), LLMCallError("timeout")],
            max_attempts=2,
        )
        with self.assertRaises(StructuredOutputError) as ctx:
            client.generate("s", "u", NavigationAssessment)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIn("model call failed", ctx.exception.error)
        self.assertIsNone(ctx.exception.last_raw)  # no output at all

    def test_candidate_set_restriction_via_extra_validator(self):
        # the generic hook the Diagnostiker (step 3.2) uses: the chosen
        # next question must be inside the Navigator's candidate set.
        run_id = new_run_id()
        turn = turn_id(1)
        outside = _valid_assessment_dict(run_id, turn)
        outside["next_question_id"] = "sq_9.9"  # not in the candidate set
        candidates = {"sq_1.1.2"}

        def check(instance):
            if instance.next_question_id not in candidates:
                raise ValueError(
                    f"next_question_id '{instance.next_question_id}' "
                    f"is not in the candidate set"
                )
            return instance

        client = self._client([json.dumps(outside)], max_attempts=1)
        with self.assertRaises(StructuredOutputError) as ctx:
            client.generate(
                "s", "u", NavigationAssessment, extra_validator=check
            )
        self.assertIn("candidate set", ctx.exception.error)

        # the same call succeeds when the choice is inside the set:
        client = self._client(
            [json.dumps(_valid_assessment_dict(run_id, turn))], max_attempts=1
        )
        result = client.generate(
            "s", "u", NavigationAssessment, extra_validator=check
        )
        self.assertEqual(result.parsed.next_question_id, "sq_1.1.2")

    def test_generation_is_deterministic_for_the_same_script(self):
        run_id = new_run_id()
        turn = turn_id(1)
        raw = json.dumps(_valid_assessment_dict(run_id, turn))
        expected = NavigationAssessment.model_validate_json(raw)
        for _ in range(2):
            client = self._client([raw])
            result = client.generate("s", "u", NavigationAssessment)
            self.assertEqual(
                result.parsed.model_dump_json(),
                expected.model_dump_json(),
            )
            self.assertEqual(len(client._caller.calls), 1)



class TestLLMSettingsFile(unittest.TestCase):
    """load_llm_settings(): the researcher-editable YAML contract."""

    def test_the_committed_settings_file_loads(self):
        # the real study_content/llm_settings.yaml must always parse:
        settings = load_llm_settings()  # the default path
        for agent in ("diagnostiker", "sprecher", "validator"):
            config = settings.resolved_for(agent)
            self.assertEqual(config.provider, "ollama")
            self.assertEqual(config.model, "llama3.2:3b")

    def test_missing_file_falls_back_to_the_built_in_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does_not_exist.yaml")
            settings = load_llm_settings(path)
        self.assertEqual(settings.resolved_for("diagnostiker").model, "llama3.2:3b")
        self.assertEqual(settings.resolved_for("diagnostiker").provider, "ollama")

    def test_empty_file_falls_back_to_the_built_in_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            settings = load_llm_settings(path)
        self.assertEqual(settings.resolved_for("sprecher").model, "llama3.2:3b")

    def test_a_malformed_yaml_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("defaults: [unclosed")
            with self.assertRaises(LLMConfigurationError):
                load_llm_settings(path)

    def test_an_unknown_field_in_the_yaml_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "typo.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("defaults:\n  provider: ollama\n  modle: llama3.2:3b\n")
            with self.assertRaises(LLMConfigurationError) as ctx:
                load_llm_settings(path)
        self.assertIn("modle", str(ctx.exception))

    def test_a_non_mapping_top_level_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("- just\n- a\n- list\n")
            with self.assertRaises(LLMConfigurationError):
                load_llm_settings(path)

    def test_partial_agent_block_merges_with_the_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "defaults:\n"
                    "  provider: ollama\n"
                    "  model: llama3.2:3b\n"
                    "validator:\n"
                    "  provider: litellm\n"
                    "  model: glm-5.3\n"
                )
            settings = load_llm_settings(path)
        validator = settings.resolved_for("validator")
        self.assertEqual(validator.provider, "litellm")
        self.assertEqual(validator.model, "glm-5.3")
        self.assertEqual(
            validator.base_url, "https://llm-api.ai-lab.uni-luebeck.de"
        )
        # the temperature came from the global default, the model block
        # of the YAML did not set it:
        self.assertEqual(validator.temperature, 0.0)
        # the other agents keep the ollama defaults:
        self.assertEqual(
            settings.resolved_for("diagnostiker").provider, "ollama"
        )



@unittest.skipUnless(_HAS_OPENAI, _SKIP_OPENAI)
class TestOpenAICaller(unittest.TestCase):
    """
    The real caller (unit level): construction and the request/
    response handling are exercised against a stubbed HTTP client —
    no network access, ever.
    """

    def _stub_client(self, content='{"ok": true}'):
        """A stub for the openai.OpenAI object (no network)."""
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=content))
                ]
            )

        return (
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                models=SimpleNamespace(
                    list=lambda: SimpleNamespace(
                        data=[SimpleNamespace(id="glm-5.3"),
                              SimpleNamespace(id="qwen3.8-27b")]
                    )
                ),
            ),
            captured,
        )

    def test_caller_constructed_for_the_institution_endpoint(self):
        import openai

        config = _ollama_config(
            provider=LLMProvider.LITELLM,
            model="glm-5.3",
            base_url="https://llm-api.ai-lab.uni-luebeck.de",
            api_key_env="LITELLM_API_KEY",
        )
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": "sk-test"}):
            caller = OpenAICaller(config)
        self.assertIsInstance(caller._client, openai.OpenAI)
        # the key never appears in the config object itself:
        self.assertEqual(caller._config.api_key_env, "LITELLM_API_KEY")

    def test_missing_key_fails_at_construction(self):
        config = _ollama_config(
            provider=LLMProvider.LITELLM,
            model="glm-5.3",
            base_url="https://llm-api.ai-lab.uni-luebeck.de",
            api_key_env="LITELLM_API_KEY",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError):
                OpenAICaller(config)

    def test_complete_returns_the_raw_content(self):
        config = _ollama_config()
        caller = OpenAICaller(config)
        stub, captured = self._stub_client()
        caller._client = stub
        out = caller.complete("system", "user", temperature=0.0, max_tokens=64)
        self.assertEqual(out, '{"ok": true}')
        self.assertEqual(captured["model"], "llama3.2:3b")
        self.assertEqual(captured["max_tokens"], 64)
        self.assertEqual(
            captured["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )
        self.assertNotIn("response_format", captured)  # json_mode off

    def test_json_mode_forwards_the_response_format(self):
        config = _ollama_config(json_mode=True)
        caller = OpenAICaller(config)
        stub, captured = self._stub_client()
        caller._client = stub
        caller.complete("s", "u", temperature=0.0, max_tokens=64)
        self.assertEqual(captured["response_format"], {"type": "json_object"})

    def test_api_error_is_wrapped_in_llm_call_error(self):
        import openai

        config = _ollama_config()
        caller = OpenAICaller(config)

        def boom(**kwargs):
            raise openai.OpenAIError("connection refused")

        caller._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=boom)
            )
        )
        with self.assertRaises(LLMCallError) as ctx:
            caller.complete("s", "u", temperature=0.0, max_tokens=64)
        self.assertIn("connection refused", str(ctx.exception))
        self.assertIn("ollama/llama3.2:3b", str(ctx.exception))

    def test_empty_response_is_a_call_error(self):
        config = _ollama_config()
        caller = OpenAICaller(config)
        stub, _ = self._stub_client()
        stub.chat.completions.create = lambda **kwargs: SimpleNamespace(
            choices=[]
        )
        caller._client = stub
        with self.assertRaises(LLMCallError) as ctx:
            caller.complete("s", "u", temperature=0.0, max_tokens=64)
        self.assertIn("no choices", str(ctx.exception))

    def test_list_models(self):
        config = _ollama_config()
        caller = OpenAICaller(config)
        stub, _ = self._stub_client()
        caller._client = stub
        self.assertEqual(
            caller.list_models(), ["glm-5.3", "qwen3.8-27b"]
        )


@unittest.skipUnless(_HAS_OPENAI, _SKIP_OPENAI)
class TestClientFactories(unittest.TestCase):
    """create_client / build_agent_clients (no network access)."""

    def test_build_agent_clients_covers_the_agent_vocabulary(self):
        settings = LLMSettings()  # all agents on the local Ollama model
        clients = build_agent_clients(settings)
        from contracts import AgentName

        self.assertEqual(set(clients), set(AgentName))
        for agent, client in clients.items():
            self.assertEqual(client.agent, agent.value)
            self.assertEqual(client.config.provider, "ollama")
            self.assertEqual(client.config.model, "llama3.2:3b")

    def test_create_client_for_the_optional_monitor(self):
        settings = LLMSettings(
            monitor=ProviderConfig(
                provider=LLMProvider.LITELLM, model="glm-5.3"
            )
        )
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": "sk-test"}):
            client = create_client(settings.resolved_for("monitor"), "monitor")
        self.assertEqual(client.agent, "monitor")
        self.assertEqual(client.config.provider, "litellm")


if __name__ == "__main__":
    unittest.main()

