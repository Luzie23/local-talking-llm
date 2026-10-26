"""
INTERVIEW GUIDE
=================
Reads one interview guide file from study_content/interviews/*.yaml and
keeps track of where the current participant is in it: which question
comes next, and when the guide is finished.

As of Phase 2 (step 2.1, RECONSTRUCTION_PLAN.md) this is a THIN WRAPPER
around the v2 interview definition model (backend/interview_definition.py):
the legacy linear YAML is mapped onto the v2 model by the adapter there
(same question order and behavior) and ALL loading is delegated to that
module. What this file keeps is only the legacy runtime cursor (which
question is "current" now, the one-follow-up cap), because the legacy
pipeline (backend/app.py) needs it. In the new protocol, those roles are
taken over by the Navigator / Supervisor (Phase 2, step 2.5 onward).

This file deliberately knows nothing about the LLM, audio, or the web
server — it only manages "what is the fixed question sequence, and where
are we in it". backend/llm_provider.py is responsible for generating the
short feedback in between questions; backend/app.py wires the two together.

Per the project's core principle "the LLM presents, it does not invent":
question wording always comes from here, verbatim, exactly as written in
the YAML file — never generated or reworded by the model.

As of Step 3 ("Constrain the LLM"), a question may optionally allow a
single, short LLM-initiated follow-up (see the "allow_followup" field per
question, and start_followup()/is_awaiting_followup_answer()/
followup_already_used() below). This is deliberately capped at exactly one
follow-up per question for now. If you need something more elaborate later
(e.g. checking whether an open-ended answer covered several required
points, and allowing more than one follow-up until it does), this class —
together with backend/llm_provider.py's parsing of the model's response —
is the place to extend; see the comment on parse_feedback_and_followup()
in llm_provider.py for more detail on that.

Used by: backend/app.py.
"""
import os
from typing import Any

import yaml

from interview_definition import (
    INTERVIEWS_DIR,
    InterviewDefinition,
    InterviewDefinitionError,
    to_v2,
    validate_definition,
)


class InterviewGuide:
    """Tracks one participant's progress through one interview guide file."""

    def __init__(self, guide_filename: str):
        """
        Load a guide file (legacy format) and map it onto the v2 interview
        definition model (Phase 2, step 2.1).

        Args:
            guide_filename: File name inside study_content/interviews/,
                e.g. "depression_scid_example.yaml".
        """
        path = os.path.join(INTERVIEWS_DIR, guide_filename)
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        if "definition_version" in data:
            raise InterviewDefinitionError(
                f"{guide_filename} uses the v2 interview definition format "
                "(top-level 'definition_version' key / hierarchical "
                "sections), but the legacy pipeline can only present legacy "
                "guides — point study_content/study_settings.yaml at a "
                "legacy guide until the new pipeline takes over "
                "(RECONSTRUCTION_PLAN.md Phase 5)"
            )
        # Phase 2 (step 2.1): ALL loading is delegated to the v2 definition
        # model. The adapter maps this legacy file onto it, keeping the
        # question order and behavior (ids are generated with the v2
        # convention sec_/qg_/sq_); `definition_version` names the
        # exact guide content (a content hash) for the audit trail.
        self.definition: InterviewDefinition = to_v2(data, guide_filename)
        validate_definition(self.definition, guide_filename, require_questions=False)

        # The legacy public surface (used by backend/app.py):
        self.guide_id = self.definition.guide_id
        self.title = self.definition.title
        # Not used yet — this is the hook for Step 3 ("constrain the LLM"),
        # which will map this label to an actual set of behavior
        # instructions (e.g. "strict_clinical" vs "flexible_conversational").
        self.behavior_profile = self.definition.behavior_profile
        self.framing_message = self.definition.framing_message
        self.closing_message = self.definition.closing_message

        # All questions across the whole hierarchy (sections > question
        # groups > single questions), flattened into one ordered list,
        # since the pilot presents the guide strictly in sequence
        # (no branching, no randomization within the guide).
        self.questions = [
            {
                "id": q.id,
                "text": q.text,
                "section_title": group.title,
                # Whether a single, short LLM-initiated follow-up is
                # allowed for this specific item (see Step 3 / the
                # "strict_clinical"/"flexible_conversational" behavior
                # profiles). Defaults to False if not set.
                "allow_followup": q.allow_followup,
            }
            for group, q in self.definition.questions_with_group()
        ]

        # Index of the question that comes next. A single fixed value is
        # used because, per the project's core principles, only one
        # participant runs through the study at a time.
        self._current_index = 0

        # Follow-up state for the CURRENT question only. Capped at one
        # follow-up per question, per the project plan ("eine kurze
        # Nachfrage"). Reset automatically whenever advance() moves on.
        self._awaiting_followup_answer = False
        self._followup_used_for_current = False
        self.pending_followup_text: str | None = None

    @property
    def definition_version(self) -> str:
        """
        The v2 definition version of the loaded guide content
        ("<guide_id>@legacy-<content hash>"), for the audit trail.
        """
        return self.definition.definition_version

    def current_question(self) -> dict | None:
        """Return the question that should be asked right now, or None if the guide is finished."""
        if self._current_index >= len(self.questions):
            return None
        return self.questions[self._current_index]

    def advance(self) -> dict | None:
        """
        Move on to the next question, after the current one (and any
        follow-up to it) has been asked and answered.

        Returns:
            The new current question, or None if the guide is now finished.
        """
        self._current_index += 1
        self._awaiting_followup_answer = False
        self._followup_used_for_current = False
        self.pending_followup_text = None
        return self.current_question()

    def is_finished(self) -> bool:
        """True once every question in the guide has been asked."""
        return self._current_index >= len(self.questions)

    def start_followup(self, followup_text: str) -> None:
        """
        Record that the model just asked a follow-up for the current
        question. The participant's next answer should then be treated as
        answering this follow-up, not the next fixed question — see
        is_awaiting_followup_answer().
        """
        self._awaiting_followup_answer = True
        self._followup_used_for_current = True
        self.pending_followup_text = followup_text

    def is_awaiting_followup_answer(self) -> bool:
        """True if the participant's next answer is a response to a follow-up, not a new fixed question."""
        return self._awaiting_followup_answer

    def followup_already_used(self) -> bool:
        """
        True if a follow-up has already been used for the current question
        — used to enforce the "at most one follow-up per question" cap.
        """
        return self._followup_used_for_current

    def reset(self) -> None:
        """Start the guide over from the first question (e.g. for a new participant)."""
        self._current_index = 0
        self._awaiting_followup_answer = False
        self._followup_used_for_current = False
        self.pending_followup_text = None
