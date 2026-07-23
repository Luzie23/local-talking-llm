"""
BEHAVIOR PROFILES
===================
Loads a researcher-editable behavior profile: which instructions the model
gets for giving feedback, and how the next fixed question is delivered
afterwards (see study_content/prompts/profiles.yaml for the two currently
defined profiles, and study_content/prompts/*.txt for the instruction texts
themselves).

Which profile is used for a given interview guide is set by that guide's
`behavior_profile` field (see study_content/interviews/*.yaml) — this file
only knows how to load a profile by name, not which one is "active".

Used by: backend/app.py, backend/llm_provider.py.
"""
import os

import yaml

PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content", "prompts"
)
PROFILES_PATH = os.path.join(PROMPTS_DIR, "profiles.yaml")

# Used when an interview guide doesn't set behavior_profile at all.
DEFAULT_PROFILE_NAME = "strict_clinical"


class BehaviorProfile:
    """One named behavior profile: instructions text + question-delivery style."""

    def __init__(self, name: str, instructions: str, question_delivery: str):
        self.name = name
        self.instructions = instructions
        # "verbatim" (attach the next question unchanged) or
        # "natural_transition" (reword the transition, keep the meaning).
        self.question_delivery = question_delivery


def _strip_comment_lines(text: str) -> str:
    """
    Remove lines that start with "#" (after leading whitespace) — these are
    researcher-facing comments in the .txt files, not part of what gets
    sent to the model.
    """
    kept_lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return "\n".join(kept_lines).strip()


def load_behavior_profile(profile_name: str | None) -> BehaviorProfile:
    """
    Load one named behavior profile by reading profiles.yaml and the
    instructions .txt file it points to.

    Args:
        profile_name: e.g. "strict_clinical" — must match a top-level key
            in study_content/prompts/profiles.yaml. If None (an interview
            guide didn't set behavior_profile), DEFAULT_PROFILE_NAME is
            used instead.

    Returns:
        A BehaviorProfile with the loaded, comment-stripped instructions text.

    Raises:
        KeyError: if profile_name isn't listed in profiles.yaml.
        FileNotFoundError: if the referenced instructions file doesn't exist.
    """
    profile_name = profile_name or DEFAULT_PROFILE_NAME

    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    if profile_name not in registry:
        raise KeyError(
            f"Behavior profile '{profile_name}' is not defined in "
            f"study_content/prompts/profiles.yaml"
        )

    entry = registry[profile_name]
    instructions_path = os.path.join(PROMPTS_DIR, entry["instructions_file"])
    with open(instructions_path, "r", encoding="utf-8") as f:
        raw_instructions = f.read()

    instructions = _strip_comment_lines(raw_instructions)
    question_delivery = entry.get("question_delivery", "verbatim")
    return BehaviorProfile(profile_name, instructions, question_delivery)
