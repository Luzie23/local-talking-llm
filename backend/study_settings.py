"""
STUDY SETTINGS LOADER
=======================
Reads study_content/study_settings.yaml — the small set of overall study
knobs the researcher can edit without touching any code (currently: the
interview language, and which interview guide file is active).

This file only loads the settings; it does not decide what to do with them
— that happens in backend/app.py and backend/interview_guide.py.

Used by: backend/app.py.
"""
import os

import yaml

STUDY_CONTENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content"
)
SETTINGS_PATH = os.path.join(STUDY_CONTENT_DIR, "study_settings.yaml")


def load_study_settings() -> dict:
    """
    Load study_content/study_settings.yaml as a plain Python dictionary.

    Returns:
        A dict of settings, e.g. {"language": "de", "interview_guide": "..."}.
        If the file is missing or empty, an empty dict is returned rather
        than raising an error, so the app can fall back to sensible defaults.
    """
    if not os.path.exists(SETTINGS_PATH):
        return {}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
