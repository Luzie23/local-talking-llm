"""
SAFETY-NET KEYWORD CHECK
==========================
A deliberately simple, transparent check: does the participant's OWN answer
(never the LLM's output) contain any of the phrases configured in
study_content/safety_keywords.yaml? If so, the frontend shows a calm,
non-blocking hint with support resources — the interview is NOT interrupted
or stopped, this only makes sure help is visible.

This is intentionally NOT an AI/LLM judgment call. A plain keyword match is
blunt (misses paraphrases, can fire on an unrelated or negated mention), but
it is auditable and predictable — a research ethics board can see exactly
what triggers it. Asking the LLM itself to judge "is this concerning" would
reintroduce the kind of clinical assessment this whole system deliberately
avoids everywhere else (see backend/llm_provider.py and the project plan).

Explicitly out of scope here: detecting other categories of concerning
content (e.g. disclosures of criminal acts). That raises separate legal/
reporting questions for the research ethics board, not a coding decision —
see the project plan's "Offene Punkte".

Used by: backend/app.py (both /api/chat and /api/chat/audio).
"""
import os

import yaml

STUDY_CONTENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content"
)
KEYWORDS_PATH = os.path.join(STUDY_CONTENT_DIR, "safety_keywords.yaml")


def _load_keywords_by_language() -> dict[str, list[str]]:
    """
    Load study_content/safety_keywords.yaml as {"de": [...], "en": [...]}.
    Missing file or empty -> {} (fails safe: check_for_safety_concern() then
    simply never matches, rather than crashing the app over a missing
    optional file).
    """
    if not os.path.exists(KEYWORDS_PATH):
        return {}
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    result = {}
    for lang, section in data.items():
        if isinstance(section, dict) and isinstance(section.get("keywords"), list):
            result[lang] = [str(k).strip().lower() for k in section["keywords"] if str(k).strip()]
    return result


_KEYWORDS_BY_LANGUAGE = _load_keywords_by_language()


def check_for_safety_concern(text: str, language_code: str) -> bool:
    """
    Returns True if any configured keyword/phrase for `language_code` appears
    in `text` (case-insensitive substring match). Called on the
    participant's own answer only, never on the LLM's output.

    If `language_code` has no keyword list configured in
    safety_keywords.yaml, this falls back to searching across ALL configured
    languages combined — better to search too broadly than to silently check
    nothing just because of a missing config entry.
    """
    if not text or not _KEYWORDS_BY_LANGUAGE:
        return False

    lowered = text.lower()
    keywords = _KEYWORDS_BY_LANGUAGE.get(language_code)
    if keywords is None:
        keywords = [kw for kws in _KEYWORDS_BY_LANGUAGE.values() for kw in kws]

    return any(keyword in lowered for keyword in keywords)
