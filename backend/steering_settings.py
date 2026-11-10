"""
STEERING SETTINGS LOADER
========================
Reads study_content/steering_settings.yaml — the researcher-editable
STEERING policy (WInf-inspired): the per-kind re-ask thresholds of the
deterministic steering reaction table (interview_navigator.
steering_decision), the explicit skip / continue / pause command
phrases, the technical-failure pause threshold, and the
participant-facing fixed texts of the failure/pause protocol.

The typed policy is contracts.SteeringConfig (every field has a
default there). This file only LOADS the settings; it does not decide
what to do with them — the Supervisor (backend/interview_supervisor.py)
applies them. The Supervisor's default is the built-in SteeringConfig
defaults (pure, no I/O); the deployment wiring (Phase 5 / app.py)
passes load_steering_settings().

Used by: the deployment wiring (later: backend/app.py).
"""
import os

import yaml
from pydantic import ValidationError

from contracts import SteeringConfig

STEERING_CONTENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content"
)
STEERING_SETTINGS_PATH = os.path.join(STEERING_CONTENT_DIR, "steering_settings.yaml")


def load_steering_settings(path: str | None = None) -> SteeringConfig:
    """
    Load study_content/steering_settings.yaml as a SteeringConfig.

    The YAML file supplies a SUBSET of the fields (all of them have
    defaults on the model); missing keys fall back to the defaults, so
    the file may contain only the knobs the researcher wants to change.

    Behavior:
      * missing or empty file -> the built-in defaults (SteeringConfig());
      * a malformed value or an unknown key -> ValueError with an
        actionable message (the policy must fail LOUDLY at load time —
        a steering policy is a study protocol, not a guess).
    """
    settings_path = path if path is not None else STEERING_SETTINGS_PATH
    if not os.path.exists(settings_path):
        return SteeringConfig()
    with open(settings_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"steering settings must be a mapping of setting -> value, "
            f"got {type(data).__name__} in {settings_path}"
        )
    try:
        return SteeringConfig(**data)
    except ValidationError as exc:
        raise ValueError(
            f"invalid steering settings in {settings_path}: {exc}"
        ) from exc