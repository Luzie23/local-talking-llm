"""
INTERVIEW GUIDE
=================
Reads one interview guide file from study_content/interviews/*.yaml and
keeps track of where the current participant is in it: which question
comes next, and when the guide is finished.

This file deliberately knows nothing about the LLM, audio, or the web
server — it only manages "what is the fixed question sequence, and where
are we in it". backend/llm_provider.py is responsible for generating the
short feedback in between questions; backend/app.py wires the two together.

Per the project's core principle "the LLM presents, it does not invent":
question wording always comes from here, verbatim, exactly as written in
the YAML file — never generated or reworded by the model.

Used by: backend/app.py.
"""
import os

import yaml

INTERVIEWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content", "interviews"
)


class InterviewGuide:
    """Tracks one participant's progress through one interview guide file."""

    def __init__(self, guide_filename: str):
        """
        Load a guide file and flatten all its questions into one ordered list.

        Args:
            guide_filename: File name inside study_content/interviews/,
                e.g. "depression_scid_example.yaml".
        """
        path = os.path.join(INTERVIEWS_DIR, guide_filename)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.guide_id = data.get("guide_id", guide_filename)
        self.title = data.get("title", "")

        # Not used yet — this is the hook for Step 3 ("constrain the LLM"),
        # which will map this label to an actual set of behavior
        # instructions (e.g. "strict_clinical" vs "flexible_conversational").
        self.behavior_profile = data.get("behavior_profile")

        self.framing_message = (data.get("framing_message") or "").strip()
        self.closing_message = (data.get("closing_message") or "").strip()

        # All questions across all sections, flattened into one ordered
        # list, since the pilot presents the guide strictly in sequence
        # (no branching, no randomization within the guide).
        self.questions = []
        for section in data.get("sections", []):
            section_title = section.get("section_title")
            for question in section.get("questions", []):
                self.questions.append({
                    "id": question["id"],
                    "text": question["text"],
                    "section_title": section_title,
                })

        # Index of the question that comes next. A single fixed value is
        # used because, per the project's core principles, only one
        # participant runs through the study at a time.
        self._current_index = 0

    def current_question(self) -> dict | None:
        """Return the question that should be asked right now, or None if the guide is finished."""
        if self._current_index >= len(self.questions):
            return None
        return self.questions[self._current_index]

    def advance(self) -> dict | None:
        """
        Move on to the next question, after the current one has been asked
        and answered.

        Returns:
            The new current question, or None if the guide is now finished.
        """
        self._current_index += 1
        return self.current_question()

    def is_finished(self) -> bool:
        """True once every question in the guide has been asked."""
        return self._current_index >= len(self.questions)

    def reset(self) -> None:
        """Start the guide over from the first question (e.g. for a new participant)."""
        self._current_index = 0
