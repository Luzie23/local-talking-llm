"""
LLM CLIENTS — THE PROVIDER LAYER BEHIND THE PHASE 1 CONTRACTS
=============================================================
RECONSTRUCTION_PLAN.md, Phase 3, step 3.1 — "the LLM agent layer":
the per-agent provider configuration, the STRICT structured JSON output
(schema-validated against the Phase 1 contracts of backend/contracts.py),
and the DETERMINISTIC FAKE LLM that is the test double for the contract
tests of step 3.5 and for the agent implementations of steps 3.2-3.4.

This module is the ONLY place of the new pipeline that talks to a
language model. The legacy pipeline (backend/app.py) keeps using
backend/llm_provider.py (LangChain-based) until the Phase 8 cutover —
the two are deliberately independent.

PROVIDERS (owner addition 2026-11-04: the plan named ollama/minimax;
the institution's LiteLLM proxy and HuggingFace are added here). All
four speak the OpenAI chat-completions wire format, so ONE client
(the 'openai' package, lazily imported) serves all of them:

  * "ollama"      — a model running fully locally on this machine
                    (the local-first data principle: real study data
                    stays on this machine). Ollama's OpenAI-compatible
                    endpoint at http://localhost:11434/v1. No key.
  * "minimax"     — the cloud API of the legacy provider
                    (backend/llm_provider.py), test-only — NOT for
                    real study data. Key env var: MINIMAX_API_KEY.
  * "litellm"     — the institution's LiteLLM proxy
                    (https://llm-api.ai-lab.uni-luebeck.de), an
                    OpenAI-compatible API: the standard OpenAI client
                    with base_url + API key changed (the models are
                    queryable via /v1/models). Key env var:
                    LITELLM_API_KEY (configurable via api_key_env).
  * "huggingface" — HuggingFace Inference Providers, exposed as an
                    OpenAI-compatible API at
                    https://api-inference.huggingface.co/v1. Key env
                    var: HF_TOKEN (configurable via api_key_env).
                    (Models the institution already hosts via LiteLLM
                    are reached through the "litellm" provider — this
                    provider is for models pulled straight from
                    HuggingFace.)

CONFIGURATION: study_content/llm_settings.yaml (researcher-editable,
like study_settings.yaml) — one partial provider block per agent plus
a `defaults` block; resolution merges provider built-ins < defaults
< per-agent entry (later wins), see ResolvedProviderConfig.from_layers.

SECRETS: API keys are NEVER stored in the YAML, never logged and never
put into the audit events — only in the environment variable named by
the resolved config's `api_key_env`. A missing key fails LOUDLY at
client construction (LLMConfigurationError), not mid-session.

STRICT STRUCTURED OUTPUT: every agent call goes through
StructuredLLMClient.generate() — the raw reply is extracted as a single
JSON object and validated with Pydantic against the Phase 1 contract
(the frozen, extra="forbid" models of contracts.py), with a bounded
retry loop (config.max_attempts); schema violations and parse failures
are fed back to the model with the concrete error. After the last
failed attempt StructuredOutputError is raised — FAIL-CLOSED: no
partially parsed result ever leaves this module.

Used by: (step 3.1) tests/test_llm_clients.py, the opt-in
tests/test_litellm_integration.py; (steps 3.2-3.4) the Diagnostiker /
Sprecher / Validator agent implementations; (step 4) the
Interventionsmonitor's optional LLM layer.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from contracts import AgentName
from study_settings import STUDY_CONTENT_DIR


# ---------------------------------------------------------------------------
# 1. Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Base for all errors of this module (never a silent fallback)."""


class LLMConfigurationError(LLMClientError):
    """
    The provider / model / key configuration is wrong or incomplete
    (missing model, missing API key env var, malformed settings file).
    Raised at configuration/construction time so it cannot happen
    mid-session.
    """


class LLMCallError(LLMClientError):
    """
    One model call failed: network error, timeout, API error, an empty
    response — or the FakeLLM ran out of scripted responses.
    """


class JSONExtractionError(LLMClientError):
    """The model's raw reply does not contain a usable JSON object."""


class StructuredOutputError(LLMClientError):
    """
    All bounded attempts failed (call, JSON extraction or schema
    validation). Fail-closed: the caller (the agent layer) decides the
    reaction (bounded revision loop, canonical fallback — Phase 3.4).
    Carries the agent, the attempt count, the raw output of the last
    SUCCESSFUL call (None when the last attempt had no output at all)
    and the error message of the last attempt.
    """

    def __init__(
        self,
        agent: str,
        attempts: int,
        last_raw: str | None,
        error: str,
    ) -> None:
        self.agent = agent
        self.attempts = attempts
        self.last_raw = last_raw
        self.error = error
        super().__init__(
            f"agent '{agent}': {attempts} attempt(s), last error: {error}"
        )


# ---------------------------------------------------------------------------
# 2. Provider configuration
# ---------------------------------------------------------------------------


class LLMProvider:
    """
    The four provider kinds. A plain class with string values on
    purpose: the settings YAML carries the lowercase names verbatim
    (unknown names fail loudly in ProviderConfig's validation, and
    adding a fifth backend later is a one-line change here).
    """

    OLLAMA = "ollama"
    MINIMAX = "minimax"
    LITELLM = "litellm"
    HUGGINGFACE = "huggingface"

    ALL: tuple[str, ...] = (OLLAMA, MINIMAX, LITELLM, HUGGINGFACE)


# The per-provider built-ins (lowest precedence layer): the wire
# endpoint, the key env var, the provider's customary default model
# and a provider-appropriate default timeout (local Ollama inference
# is slow — the cloud endpoints are fast but busy).
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # Ollama needs no key
        # the local model of this project's reference setup — the
        # per-user setting point is the `defaults` block of
        # study_content/llm_settings.yaml (see `ollama list`):
        "model": "llama3.2:3b",
        "timeout_s": 300.0,
    },
    LLMProvider.MINIMAX: {
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7",
        "timeout_s": 120.0,
    },
    LLMProvider.LITELLM: {
        "base_url": "https://llm-api.ai-lab.uni-luebeck.de",
        "api_key_env": "LITELLM_API_KEY",
        # deployment-specific — the institution's model names must be
        # set in study_content/llm_settings.yaml (see the AVAILABLE
        # MODELS list there: deepseek-v4-flash, glm-5.3, glm-5.3-flash,
        # gemma-4-26b-a4b-it, qwen3.8-27b, gpt-oss-120b,
        # qwen3-vl-8b-instruct, bge-m3 [embeddings, not chat]):
        "model": None,
        "timeout_s": 120.0,
    },
    LLMProvider.HUGGINGFACE: {
        "base_url": "https://api-inference.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        # which HF model to call (e.g. "Qwen/Qwen3-VL-8B-Instruct") —
        # must be set in study_content/llm_settings.yaml:
        "model": None,
        "timeout_s": 120.0,
    },
}

# Global fallbacks (lowest precedence overall) for the call parameters
# that have no per-provider meaning.
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,  # structured agent outputs: deterministic
    "timeout_s": 120.0,
    "max_attempts": 3,  # total attempts per structured call
    "max_output_tokens": 1024,
    "retry_backoff_s": 0.5,
    "json_mode": False,  # prompt-driven JSON; provider-side is opt-in
}

# The settings fields — the merge order of
# ResolvedProviderConfig.from_layers, later wins:
# provider built-ins < LLMSettings.defaults < the per-agent entry.
_CONFIG_FIELDS: tuple[str, ...] = (
    "model",
    "temperature",
    "timeout_s",
    "max_attempts",
    "max_output_tokens",
    "retry_backoff_s",
    "base_url",
    "api_key_env",
    "json_mode",
)


class ProviderConfig(BaseModel):
    """
    One PARTIAL provider specification (any field may be left out —
    None). Used both as the per-agent block and as the `defaults`
    block of study_content/llm_settings.yaml. Resolution into a
    complete ResolvedProviderConfig happens in
    ResolvedProviderConfig.from_layers; unknown fields are rejected —
    a typo must not silently configure nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str | None = Field(
        default=None,
        description="one of LLMProvider.ALL (ollama/minimax/litellm/huggingface)",
    )
    model: str | None = Field(
        default=None,
        description="the model name on that provider (deployment-specific)",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeout_s: float | None = Field(default=None, gt=0.0)
    max_attempts: int | None = Field(
        default=None,
        ge=1,
        description="total attempts per structured call (first call + retries)",
    )
    max_output_tokens: int | None = Field(default=None, ge=1)
    retry_backoff_s: float | None = Field(
        default=None,
        ge=0.0,
        description="pause between two attempts (0 in the tests)",
    )
    base_url: str | None = Field(
        default=None,
        description="overrides the provider's built-in endpoint",
    )
    api_key_env: str | None = Field(
        default=None,
        description=(
            "NAME of the environment variable holding the API key "
            "(never the key itself)"
        ),
    )
    json_mode: bool | None = Field(
        default=None,
        description="ask the provider for JSON output if it supports it",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_provider_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("provider") is not None:
            provider = data["provider"]
            if provider not in LLMProvider.ALL:
                raise ValueError(
                    f"unknown provider '{provider}' "
                    f"(one of: {', '.join(LLMProvider.ALL)})"
                )
        return data


class ResolvedProviderConfig(BaseModel):
    """
    One COMPLETE, resolved provider configuration (every field
    filled, ranges checked) — the input of the OpenAICaller and the
    StructuredLLMClient. Built only by from_layers, so the merge order
    stays in one place.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_s: float = Field(gt=0.0)
    max_attempts: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    retry_backoff_s: float = Field(ge=0.0)
    base_url: str
    api_key_env: str | None = None
    json_mode: bool = False

    @classmethod
    def from_layers(
        cls,
        defaults: ProviderConfig,
        entry: ProviderConfig | None,
        *,
        agent_label: str = "",
    ) -> ResolvedProviderConfig:
        """
        Merge the three configuration layers (later wins): provider
        built-ins < `defaults` < the per-agent `entry`.

        Raises LLMConfigurationError when the provider or the model
        cannot be determined (fail loud, at configuration time).
        """
        entry = entry if entry is not None else ProviderConfig()
        # the provider is decided first (the built-ins are keyed by it):
        provider = (
            entry.provider
            if entry.provider is not None
            else defaults.provider
        )
        if provider is None:
            raise LLMConfigurationError(
                f"agent '{agent_label}': no provider configured — set "
                f"`provider` (one of: {', '.join(LLMProvider.ALL)}) in the "
                f"`defaults` block or the agent's block of "
                f"study_content/llm_settings.yaml"
            )
        builtins = PROVIDER_DEFAULTS[provider]
        merged: dict[str, Any] = {}
        for field in _CONFIG_FIELDS:
            value: Any = getattr(entry, field)
            if value is None:
                value = getattr(defaults, field)
            if value is None:
                value = builtins.get(field)
            if value is None:
                value = _GLOBAL_DEFAULTS.get(field)
            merged[field] = value
        model: str | None = merged["model"]
        if not model:
            raise LLMConfigurationError(
                f"agent '{agent_label}': no model configured for "
                f"provider '{provider}' — set `model` in the `defaults` "
                f"block or the agent's block of "
                f"study_content/llm_settings.yaml"
            )
        if not merged["base_url"]:
            raise LLMConfigurationError(
                f"agent '{agent_label}': no endpoint (base_url) "
                f"resolvable for provider '{provider}'"
            )
        try:
            return cls(
                provider=provider,
                model=model,
                temperature=merged["temperature"],
                timeout_s=merged["timeout_s"],
                max_attempts=merged["max_attempts"],
                max_output_tokens=merged["max_output_tokens"],
                retry_backoff_s=merged["retry_backoff_s"],
                base_url=merged["base_url"],
                api_key_env=merged["api_key_env"],
                json_mode=merged["json_mode"],
            )
        except ValidationError as exc:
            raise LLMConfigurationError(
                f"agent '{agent_label}': invalid resolved "
                f"configuration: {exc}"
            ) from exc


class LLMSettings(BaseModel):
    """
    The parsed study_content/llm_settings.yaml: a `defaults` block
    plus one optional block per pipeline agent (the
    Interventionsmonitor is included already — its LLM layer arrives
    in Phase 4 and is not part of the AgentName vocabulary yet).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # the local-first fallback when the file is missing entirely:
    # every agent talks to the local Ollama model llama3.2:3b (the
    # reference setup of this project). The real configuration point
    # is study_content/llm_settings.yaml, where each user sets the
    # model they have installed locally.
    defaults: ProviderConfig = ProviderConfig(
        provider=LLMProvider.OLLAMA, model="llama3.2:3b"
    )
    diagnostiker: ProviderConfig | None = None
    sprecher: ProviderConfig | None = None
    validator: ProviderConfig | None = None
    monitor: ProviderConfig | None = None

    _KNOWN_AGENTS: tuple[str, ...] = (
        "diagnostiker",
        "sprecher",
        "validator",
        "monitor",
    )

    def entry_for(self, agent_label: str) -> ProviderConfig | None:
        """The raw per-agent block (None when the agent inherits defaults)."""
        if agent_label not in self._KNOWN_AGENTS:
            raise LLMConfigurationError(
                f"unknown agent '{agent_label}' "
                f"(one of: {', '.join(self._KNOWN_AGENTS)})"
            )
        return getattr(self, agent_label)

    def resolved_for(self, agent_label: str) -> ResolvedProviderConfig:
        """The complete configuration of one agent (three-layer merge)."""
        return ResolvedProviderConfig.from_layers(
            self.defaults, self.entry_for(agent_label), agent_label=agent_label
        )


SETTINGS_FILE_NAME = "llm_settings.yaml"
DEFAULT_SETTINGS_PATH = os.path.join(STUDY_CONTENT_DIR, SETTINGS_FILE_NAME)


def load_llm_settings(path: str | os.PathLike | None = None) -> LLMSettings:
    """
    Load (and validate) the LLM settings file. A MISSING file yields
    the built-in defaults (local Ollama) rather than an error — the
    app can fall back to sensible defaults, the same contract
    study_settings.py has. A file that EXISTS but does not validate
    raises LLMConfigurationError (a broken settings file must not
    silently reconfigure the agents).
    """
    settings_path = str(path if path is not None else DEFAULT_SETTINGS_PATH)
    if not os.path.exists(settings_path):
        return LLMSettings()
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise LLMConfigurationError(
            f"{settings_path}: invalid YAML: {exc}"
        ) from exc
    if data is None:
        return LLMSettings()
    if not isinstance(data, dict):
        raise LLMConfigurationError(
            f"{settings_path}: the top level must be a mapping of settings"
        )
    try:
        return LLMSettings.model_validate(data)
    except ValidationError as exc:
        raise LLMConfigurationError(
            f"{settings_path}: invalid LLM settings: {exc}"
        ) from exc


def resolve_api_key(config: ResolvedProviderConfig) -> str | None:
    """
    The API key for a resolved config — read from the environment
    variable named by config.api_key_env. Ollama (api_key_env None)
    needs no key and yields None (the OpenAI client gets a dummy).
    A REQUIRED but missing/empty key raises LLMConfigurationError —
    keys are never stored in files, so the environment is the only
    source.
    """
    env_var = config.api_key_env
    if env_var is None:
        return None
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMConfigurationError(
            f"provider '{config.provider}' requires the API key "
            f"environment variable {env_var} to be set (keys are never "
            f"stored in files — export {env_var} in the shell/venv)"
        )
    return key


# ---------------------------------------------------------------------------
# 3. The callers (one chat completion -> raw text)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeLLMCall:
    """One call the FakeLLM served (recorded for the tests' assertions)."""

    system_prompt: str
    user_prompt: str
    temperature: float
    max_tokens: int | None


class LLMCaller(Protocol):
    """
    The minimal surface one LLM must provide for a structured agent
    call: one chat completion, raw text out, LLMCallError on any
    transport/API failure. OpenAICaller implements it for the four
    real providers; FakeLLM implements it deterministically for the
    tests.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """One chat completion. Raises LLMCallError when it fails."""
        ...


class OpenAICaller:
    """
    The real caller: one OpenAI-compatible chat completion per
    complete() call — the 'openai' package (lazily imported, so this
    module still imports when the package is absent and only the
    FakeLLM is needed) pointed at the provider's base_url. The client
    is created with max_retries=0: the bounded, auditable retry logic
    belongs to StructuredLLMClient (Phase 3.1), not to the HTTP layer.
    """

    # the OpenAI client requires a non-empty key; Ollama ignores it.
    _DUMMY_API_KEY = "local-ollama"

    def __init__(self, config: ResolvedProviderConfig) -> None:
        try:
            import openai
        except ImportError as exc:
            raise LLMConfigurationError(
                "the 'openai' package is required for the provider "
                f"'{config.provider}' — install it with "
                f"'pip install openai'"
            ) from exc
        self._openai = openai
        self._config = config
        self._client = openai.OpenAI(
            api_key=resolve_api_key(config) or self._DUMMY_API_KEY,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=0,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """One chat completion -> the raw text of the first choice."""
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._openai.OpenAIError as exc:
            raise LLMCallError(
                f"{self._config.provider}/{self._config.model}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMCallError(
                f"{self._config.provider}/{self._config.model}: "
                f"empty response (no choices)"
            )
        content = getattr(choices[0].message, "content", None)
        if content is None:
            raise LLMCallError(
                f"{self._config.provider}/{self._config.model}: "
                f"response without content"
            )
        return content

    def list_models(self) -> list[str]:
        """
        The model names the provider currently serves (its /v1/models
        endpoint) — for the smoke tests and for finding the right
        model name in study_content/llm_settings.yaml.
        """
        try:
            response = self._client.models.list()
        except self._openai.OpenAIError as exc:
            raise LLMCallError(
                f"{self._config.provider}/{self._config.model}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return sorted(str(m.id) for m in response.data)


class FakeLLM:
    """
    The deterministic fake LLM (the step 3.1 test double).

    It is an LLMCaller: it serves SCRIPTED raw responses one by one
    and performs no I/O — so the REAL client code path (the JSON
    extraction, the Pydantic schema validation, the bounded retry
    loop, the fail-closed behavior) runs unmodified in the tests.
    A scripted Exception (e.g. LLMCallError("timeout")) simulates a
    failed call; a script that is exhausted raises LLMCallError.
    Every call is recorded in .calls for assertions.
    """

    def __init__(self, raw_responses: Sequence[str | Exception] = ()) -> None:
        self._script: list[str | Exception] = list(raw_responses)
        self._next = 0
        self.calls: list[FakeLLMCall] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        self.calls.append(
            FakeLLMCall(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if self._next >= len(self._script):
            raise LLMCallError("FakeLLM: scripted responses exhausted")
        item = self._script[self._next]
        self._next += 1
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def responses_left(self) -> int:
        """How many scripted responses are still unserved."""
        return len(self._script) - self._next

    def reset(self, raw_responses: Sequence[str | Exception] = ()) -> None:
        """Replace the script and clear the call record."""
        self._script = list(raw_responses)
        self._next = 0
        self.calls = []


def build_caller(config: ResolvedProviderConfig) -> LLMCaller:
    """The real caller for a resolved config (all four providers)."""
    if config.provider not in LLMProvider.ALL:
        raise LLMConfigurationError(
            f"unknown provider '{config.provider}' "
            f"(one of: {', '.join(LLMProvider.ALL)})"
        )
    return OpenAICaller(config)


# ---------------------------------------------------------------------------
# 4. The structured JSON generation (strict, bounded, fail-closed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMGeneration:
    """
    The result of one StructuredLLMClient.generate() call: the
    VALIDATED contract instance plus the raw model output of the
    succeeded attempt (the audit trail keeps raw + parsed — the
    AgentOutputEvent of contracts.py) and the attempts it took (1 =
    first try; the measure of the structured-output reliability risk
    #1 of the plan).
    """

    parsed: BaseModel
    raw: str
    attempts: int


_CODE_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_code_fence(raw: str) -> str:
    """The content of the first markdown code fence, or the raw reply."""
    match = _CODE_FENCE.search(raw)
    if match is None:
        return raw
    return match.group(1).strip()


def _first_balanced_object(text: str) -> str:
    """
    The first top-level {...} block of the text (string-aware: braces
    inside JSON string values do not count). Raises JSONExtractionError
    when the reply does not contain a complete object.
    """
    start = text.find("{")
    if start == -1:
        raise JSONExtractionError("the reply does not contain a '{' at all")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise JSONExtractionError(
        "the reply contains an unbalanced '{' — no complete JSON object"
    )


def extract_json_object(raw: str) -> str:
    """
    The JSON object the model produced, extracted from its raw reply.

    Tolerates: (a) the object alone, (b) the object wrapped in a
    ```json ... ``` (or plain ```) fence, (c) a short prose lead-in
    before the object. Anything stricter than that is the model's
    problem: this function fails with JSONExtractionError and the
    bounded retry loop (with the concrete error fed back) deals with
    it.
    """
    candidate = _strip_code_fence(raw)
    json_text = _first_balanced_object(candidate)
    try:
        json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(
            f"the reply's JSON object does not parse: {exc}"
        ) from exc
    return json_text


def _describe_error(exc: Exception) -> str:
    """The one-line error message fed back to the model (and the audit)."""
    if isinstance(exc, ValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        return f"field '{location or '?'}': {first.get('msg', exc)}"
    return str(exc)


_CORRECTION_NOTE = (
    "\n\nIMPORTANT: your previous reply was unusable. Problem: {error}\n"
    "Reply again with ONLY the JSON object that matches the required "
    "schema — no prose, no markdown fences, no comments."
)




class StructuredLLMClient:
    """
    One LLM behind the Phase 1 contracts: it takes a system prompt, a
    user prompt and the expected contract schema and returns the
    VALIDATED contract instance (LLMGeneration). STRICT by design:

      * the raw reply must contain one JSON object (fences/prose
        tolerated, see extract_json_object);
      * the object must validate against the schema — the Phase 1
        contracts are frozen and extra="forbid", so unknown fields,
        wrong types and wrong enum values fail;
      * an optional extra_validator (a pipeline constraint the
        contract itself cannot express — e.g. the Diagnostiker's
        next_question_id must be inside the Navigator's candidate
        set, step 3.2) is applied after the schema check;
      * a bounded retry loop (config.max_attempts total attempts): a
        failed call, a non-JSON reply or a schema violation triggers
        the next attempt; parse/schema failures are fed back to the
        model with the concrete error (the correction note);
      * after the last failed attempt StructuredOutputError is raised
        — FAIL-CLOSED, no partial result ever leaves generate().
    """

    def __init__(
        self,
        agent_label: str,
        config: ResolvedProviderConfig,
        caller: LLMCaller | None = None,
    ) -> None:
        self._agent = agent_label
        self._config = config
        # an injected caller is the test seam (the FakeLLM); otherwise
        # the real OpenAI-compatible caller for the configured provider:
        self._caller: LLMCaller = (
            caller if caller is not None else build_caller(config)
        )

    @property
    def agent(self) -> str:
        """The agent label this client serves (diagnostiker/sprecher/...)."""
        return self._agent

    @property
    def config(self) -> ResolvedProviderConfig:
        """The resolved provider configuration (provider, model, ...)."""
        return self._config

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        extra_validator: Callable[[BaseModel], BaseModel] | None = None,
    ) -> LLMGeneration:
        """
        One structured agent call. See the class docstring for the
        strictness and retry contract. Raises StructuredOutputError
        (fail-closed) when all attempts failed.
        """
        last_error = "no attempt was made"
        last_raw: str | None = None
        current_user = user_prompt
        attempts = 0
        for attempt in range(1, self._config.max_attempts + 1):
            attempts = attempt
            try:
                raw = self._caller.complete(
                    system_prompt,
                    current_user,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_output_tokens,
                )
            except LLMCallError as exc:
                last_error = f"model call failed: {exc}"
            else:
                last_raw = raw
                try:
                    json_text = extract_json_object(raw)
                    instance = schema.model_validate_json(json_text)
                    if extra_validator is not None:
                        instance = extra_validator(instance)
                    return LLMGeneration(
                        parsed=instance, raw=raw, attempts=attempts
                    )
                except (
                    JSONExtractionError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    last_error = _describe_error(exc)
            if attempt < self._config.max_attempts:
                if self._config.retry_backoff_s > 0:
                    time.sleep(self._config.retry_backoff_s)
                current_user = (
                    user_prompt + _CORRECTION_NOTE.format(error=last_error)
                )
        raise StructuredOutputError(
            agent=self._agent,
            attempts=attempts,
            last_raw=last_raw,
            error=last_error,
        )


def create_client(
    config: ResolvedProviderConfig, agent_label: str
) -> StructuredLLMClient:
    """
    One REAL client for one agent (the caller is built from the
    resolved config — the OpenAI-compatible endpoint, so constructing
    performs no network access).
    """
    return StructuredLLMClient(agent_label, config)


def build_agent_clients(
    settings: LLMSettings,
) -> dict[AgentName, StructuredLLMClient]:
    """
    One client per agent of the AgentName vocabulary (diagnostiker,
    sprecher, validator) — the wiring the agent implementations of
    steps 3.2-3.4 consume. The optional monitor client (Phase 4) is
    not part of the AgentName vocabulary; build it with create_client(
    settings.resolved_for("monitor"), "monitor").
    """
    return {
        agent: create_client(settings.resolved_for(agent.value), agent.value)
        for agent in AgentName
    }
