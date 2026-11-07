"""
INTERVIEW GUIDE
=================
Reads one interview guide file from study_content/interviews/*.yaml and
keeps the explicit INTERVIEW STATE for the current participant: which item
is active, what the participant has said about it so far, how much of the
clarification budget is used, and the finished per-item records.

This file deliberately knows nothing about the LLM, audio, or the web
server. It is the state store that the Orchestrator
(backend/agents/interview_orchestrator.py) reads and changes. The
Navigation Agent only PROPOSES actions; the Orchestrator validates them and
then calls the state-changing methods here (begin_clarification(),
finish_item(), ...). Nothing in this file decides what an answer means.

Per the project's core principle "the LLM presents, it does not invent":
question wording always comes from the YAML file. The Question Agent may
explain or reformulate the CURRENT item, but never adds new items.

Item fields (all optional except id/text — see the example guide's header):
  id, text, criterion_id     as before
  criterion                  human-readable description of what the item
                             assesses (passed to the Judgment/Question Agents)
  rating_scheme              "none" (default) or "scid5pd" (? / 0 / 1 / 2)
  rating_anchor              the source's threshold anchor, e.g. what a "2"
                             requires; only used with a rating scheme
  probes                     source follow-up probes for this item (context
                             for the Question Agent; not asked automatically)
  max_clarifications         per-item clarification budget; falls back to the
                             guide's default_max_clarifications (default 1).
                             0 means "do not clarify, move on".
  next_step                  id of the item that follows, or "END". Validated
                             at load time. Missing = the next item in order.

SCID-5-PD support (see study_content/interviews/scid5pd_de.yaml header):
  criterion_code             source code, e.g. "PD27" (shared by questions of one criterion)
  intro                      source introduction shown before the item
  ask: false                 item not asked by the chatbot (observation /
                             clinician-only / derived / same_as / gate);
                             recorded automatically when reached
  gate                       source-defined decision point:
                             {criterion_codes, min_rated_2, if_met, if_not_met}
  cannot_answer_moves_on     source rule "IF CAN'T ANSWER, MOVE ON"
  safety_relevant            item concerns suicidality/self-harm (flag in record)

Guide-level fields added for the agent workflow:
  default_max_clarifications  project rule, default 1
  skip_phrases                exact explicit skip commands (deterministic,
                              works even if the LLM fails)
  interaction_messages        fixed texts for skips/technical problems

Used by: backend/app.py, backend/agents/interview_orchestrator.py.
"""
import os
import copy
import yaml

INTERVIEWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_content", "interviews"
)

# Value used by next_step to mark the end of the interview.
END_MARKER = "END"

# Rating schemes the application understands. "scid5pd" is the coding of
# the SCID-5-PD (source-defined): ? = inadequate information, 0 = absent,
# 1 = subthreshold, 2 = threshold.
RATING_SCHEMES = {
    "none": [],
    "scid5pd": ["?", "0", "1", "2"],
}

# Fallback fixed texts (project-specific wording, not SCID). A guide can
# override any of them under "interaction_messages". They are deliberately
# fixed rather than LLM-generated so that a refusal can never be answered
# with persuasion, and so that they still work when the model fails.
DEFAULT_MESSAGES = {
    "de": {
        "skip_refused": "Das ist völlig in Ordnung, wir lassen diese Frage aus.",
        "skip_unresolved": "Danke. Wir gehen zur nächsten Frage weiter.",
        "skip_cannot_answer": "Das ist in Ordnung. Wir gehen zur nächsten Frage weiter.",
        "system_error": "Entschuldigung, bei der Verarbeitung gab es ein technisches Problem. "
                        "Könnten Sie Ihre Antwort bitte noch einmal senden?",
        "paused": "Es gibt gerade ein anhaltendes technisches Problem. Das Interview ist pausiert. "
                  "Bitte wenden Sie sich an das Studienteam.",
        "acknowledge": "Danke für Ihre Antwort.",
    },
    "en": {
        "skip_refused": "That is completely fine, we will leave this question out.",
        "skip_unresolved": "Thank you. Let's move on to the next question.",
        "skip_cannot_answer": "That's fine. Let's move on to the next question.",
        "system_error": "Sorry, there was a technical problem processing your answer. "
                        "Could you please send it again?",
        "paused": "There is an ongoing technical problem. The interview is paused. "
                  "Please contact the study team.",
        "acknowledge": "Thank you for your answer.",
    },
}

DEFAULT_SKIP_PHRASES = {
    "de": [
        "überspringen",
        "frage überspringen",
        "überspringe diese frage",
        "nächste frage",
        "weiter",
        "keine angabe",
    ],
    "en": ["skip", "skip this", "skip this question", "next question", "pass"],
}


class GuideConfigError(ValueError):
    """Raised when a guide file is incomplete or inconsistent (e.g. an
    unfilled template item or a next_step pointing to a missing item)."""


class InterviewGuide:
    """Tracks one participant's progress through one interview guide file."""

    def __init__(self, guide_filename: str):
        """
        Load a guide file, flatten its items into one ordered list and
        validate the configured transitions.

        Args:
            guide_filename: File name inside study_content/interviews/,
                e.g. "depression_scid_example.yaml".

        Raises:
            GuideConfigError: if an item has no question text (e.g. an
                unfilled SCID template) or a next_step target does not exist.
        """
        path = os.path.join(INTERVIEWS_DIR, guide_filename)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.guide_id = data.get("guide_id", guide_filename)
        self.title = data.get("title", "")
        self.behavior_profile = data.get("behavior_profile")

        #emsag 18.10.2026 extra changes:
        # Optional metadata for future interview-guide formats.
        # Defaults preserve compatibility with existing YAML files.
        self.guide_version = data.get("guide_version", "1.0")
        self.language = data.get("language", "de")
        self.interview_type = data.get("interview_type", "structured")
        self.navigation_mode = data.get("navigation_mode", "sequential")
        #emsag 18.10.2026 extra changes:

        # Content provenance label (e.g. "team-written mock, NOT SCID").
        # Copied into every session record so that results from mock content
        # can never be mistaken for SCID results.
        self.content_source = data.get("content_source", "unspecified")

        self.framing_message = (data.get("framing_message") or "").strip()
        self.closing_message = (data.get("closing_message") or "").strip()

        # Project rule (not a SCID rule): one clarification/reformulation
        # attempt per item unless the guide says otherwise.
        self.default_max_clarifications = max(0, int(data.get("default_max_clarifications", 1)))

        lang = self.language if self.language in DEFAULT_MESSAGES else "de"
        self.messages = dict(DEFAULT_MESSAGES[lang])
        self.messages.update(data.get("interaction_messages") or {})
        phrases = data.get("skip_phrases") or DEFAULT_SKIP_PHRASES.get(lang, [])
        self.skip_phrases = [str(p).strip().lower() for p in phrases if str(p).strip()]

        self.questions = []
        for section in data.get("sections", []):
            section_title = section.get("section_title")
            for question in section.get("questions", []):
                text = (question.get("text") or "").strip()
                ask = bool(question.get("ask", True))
                if ask and not text:
                    raise GuideConfigError(
                        f"Item {question.get('id')!r} in {guide_filename} has no question text. "
                        "Template items must be filled from the licensed source before use."
                    )
                rating_scheme = question.get("rating_scheme", data.get("rating_scheme", "none")) or "none"
                if rating_scheme not in RATING_SCHEMES:
                    raise GuideConfigError(
                        f"Item {question.get('id')!r}: unknown rating_scheme {rating_scheme!r}."
                    )
                self.questions.append({
                    "id": question["id"],
                    "text": text,
                    "section_title": section_title,
                    # Legacy single-follow-up switch (kept for compatibility;
                    # the agent workflow uses the clarification budget below).
                    "allow_followup": bool(question.get("allow_followup", False)),

                    # EMSAG 18.10.2026 Optional fields for future SCID-compatible guides.
                    "criterion_id": question.get("criterion_id", question["id"]),
                    # Clarification is allowed by default; the budget decides
                    # how often (max_clarifications: 0 disables it).
                    "allow_clarification": bool(question.get("allow_clarification", True)),
                    "max_clarifications": max(0, int(question.get(
                        "max_clarifications", self.default_max_clarifications))),
                    "next_step": question.get("next_step"),
                    # EMSAG 18.10.2026 Optional fields for future SCID-compatible guides.

                    "criterion": (question.get("criterion") or "").strip(),
                    "rating_scheme": rating_scheme,
                    "rating_anchor": (question.get("rating_anchor") or "").strip(),
                    "probes": [str(p).strip() for p in (question.get("probes") or []) if str(p).strip()],

                    # --- SCID-5-PD support -------------------------------------
                    # Source criterion code (e.g. "PD27"); several questions can
                    # share one code when the source asks several questions
                    # for one criterion.
                    "criterion_code": question.get("criterion_code"),
                    # Source introduction sentence shown once before the item.
                    "intro": (question.get("intro") or "").strip(),
                    # ask: false = item is NOT asked by the chatbot (observation
                    # items, clinician-only exclusion criteria, derived ratings,
                    # source decision points). It is processed automatically
                    # when reached, see _process_non_interactive().
                    "ask": ask,
                    "not_asked_reason": question.get("not_asked_reason"),
                    "note": (question.get("note") or "").strip(),
                    "same_as": question.get("same_as"),
                    "derived_any_rated_2_of": list(question.get("derived_any_rated_2_of") or []),
                    "gate": question.get("gate"),
                    # Source rule "IF CAN'T ANSWER, MOVE ON" for this item.
                    "cannot_answer_moves_on": bool(question.get("cannot_answer_moves_on", False)),
                    # Item touches suicidality/self-harm: flagged in the record
                    # for the clinician (the keyword safety check runs anyway).
                    "safety_relevant": bool(question.get("safety_relevant", False)),
                    # General guidance for ratings (e.g. SCID-5-PD p. 5-6),
                    # given to the Judgment Agent for rated items.
                    "rating_guidance": (data.get("rating_guidance") or "").strip(),
                })

        # Transitions are validated once at load time so that a typo in
        # next_step fails loudly here instead of silently ending or
        # misrouting an interview later.
        self._index_by_id = {q["id"]: i for i, q in enumerate(self.questions)}
        if len(self._index_by_id) != len(self.questions):
            raise GuideConfigError(f"Duplicate item ids in {guide_filename}.")
        for q in self.questions:
            target = q["next_step"]
            if target not in (None, END_MARKER) and target not in self._index_by_id:
                raise GuideConfigError(
                    f"Item {q['id']!r}: next_step {target!r} does not exist in {guide_filename}."
                )
            gate = q["gate"]
            if gate:
                for key in ("if_met", "if_not_met"):
                    if gate.get(key) not in (END_MARKER,) and gate.get(key) not in self._index_by_id:
                        raise GuideConfigError(
                            f"Gate {q['id']!r}: {key} target {gate.get(key)!r} does not exist.")
                if not gate.get("criterion_codes") or int(gate.get("min_rated_2", 0)) < 1:
                    raise GuideConfigError(f"Gate {q['id']!r} needs criterion_codes and min_rated_2.")
            if q["same_as"] and q["same_as"] not in self._index_by_id:
                raise GuideConfigError(f"Item {q['id']!r}: same_as {q['same_as']!r} does not exist.")
        
        # EMSAG: 02.11.2026
        # Preserve the original SCID question configuration.
        # Temporary participant-specific changes must not affect later sessions.
        self._original_questions = copy.deepcopy(self.questions)
        # EMSAG: 02.11.2026

        self.reset()

    # ------------------------------------------------------------------
    # Current position
    # ------------------------------------------------------------------
    def current_question(self) -> dict | None:
        """Return the item that should be asked right now, or None if the guide is finished."""
        if self._current_index >= len(self.questions):
            return None
        return self.questions[self._current_index]

    def is_finished(self) -> bool:
        """True once the interview has reached its end (all items or next_step END)."""
        return self._current_index >= len(self.questions)

    def next_item_id(self) -> str | None:
        """
        The configured successor of the current item: its next_step if set,
        otherwise the next item in file order. Returns None for the end.
        Used by the Orchestrator to validate a proposed advance/branch.
        """
        question = self.current_question()
        if question is None:
            return None
        target = question.get("next_step")
        if target == END_MARKER:
            return None
        if target:
            return target
        nxt = self._current_index + 1
        return self.questions[nxt]["id"] if nxt < len(self.questions) else None

    def has_item(self, item_id: str) -> bool:
        """True if item_id exists in this guide (transition validation)."""
        return item_id in self._index_by_id

    def advance(self) -> dict | None:
        """
        Move to the configured next item (next_step, else file order) and
        reset all per-item state.

        Returns:
            The new current item, or None if the guide is now finished.
        """
        return self._move_to(self.next_item_id())

    def _move_to(self, item_id: str | None) -> dict | None:
        """
        Jump to item_id (None = end of interview), reset per-item state and
        automatically process any items the chatbot does not ask (see
        _process_non_interactive()). Returns the next item to ASK, or None.
        """
        self._current_index = self._index_by_id[item_id] if item_id else len(self.questions)
        self._reset_item_state()
        self._skip_non_interactive()
        return self.current_question()

    def _skip_non_interactive(self) -> None:
        """Record and pass over consecutive non-interactive items."""
        guard = 0

        while self.current_question() is not None and not self.current_question()["ask"]:
            item = self.current_question()

            # EMSAG: 02.11.2026
            # Reuse Q44's question presentation when SZD_A5 was not assessed
            # under Schizotypal PD. Keep SZD_A5's own criterion identity.
            if item.get("same_as"):
                source_id = item["same_as"]
                source_record = next(
                    (r for r in self.records if r.get("item_id") == source_id),
                    None,
                )

                if not source_record or source_record.get("status") != "answered":
                    source_item = self.questions[self._index_by_id[source_id]]

                    item["ask"] = True
                    item["text"] = source_item["text"]
                    item["intro"] = source_item.get("intro", "")
                    item["probes"] = list(source_item.get("probes", []))
                    item["rating_anchor"] = source_item.get("rating_anchor", "")
                    item["rating_scheme"] = source_item.get("rating_scheme", "scid5pd")
                    break

            record, target = self._process_non_interactive(item)
            self.records.append(record)
            self._current_index = self._index_by_id[target] if target else len(self.questions)
            self._reset_item_state()

            guard += 1
            if guard > len(self.questions):
                raise GuideConfigError("Non-interactive items form a loop.")

    def _successor_of(self, item: dict) -> str | None:
        """Configured successor of an item (next_step or file order; None = end)."""
        target = item.get("next_step")
        if target == END_MARKER:
            return None
        if target:
            return target
        nxt = self._index_by_id[item["id"]] + 1
        return self.questions[nxt]["id"] if nxt < len(self.questions) else None

    def criterion_has_rating_2(self, criterion_code: str) -> bool:
        """
        True if any answered question recorded for criterion_code has a
        provisional rating of "2". Several SCID questions can belong to one
        criterion; the criterion is treated as "2" if one of its questions
        established the threshold ("?" and skipped items never count).
        """
        for rec in self.records:
            if rec.get("criterion_code") == criterion_code and rec.get("status") == "answered":
                if (rec.get("provisional_rating") or {}).get("value") == "2":
                    return True
        return False

    def _process_non_interactive(self, item: dict) -> tuple[dict, str | None]:
        """
        Build the record for an item the chatbot does not ask and return
        (record, next item id).

        * gate: SOURCE-DEFINED decision point (e.g. SCID-5-PD Antisocial PD
        "at least two Conduct Disorder criteria rated 2"). Evaluated on the
        PROVISIONAL chatbot ratings; the counted criteria are stored in the
        record so the clinician can verify the routing.
        * derived: rating derived from other criteria as stated in the source.
        * same_as / observation / clinician: recorded as
        "not assessed by chatbot – clinician rating required".
        """
        record = {
            "item_id": item["id"],
            "criterion_id": item.get("criterion_id"),
            "criterion_code": item.get("criterion_code"),
            "criterion": item.get("criterion"),
            "status": "not_asked",
            "not_asked_reason": item.get("not_asked_reason"),
            "note": item.get("note"),
        }
        gate = item.get("gate")
        if gate:
            codes = gate["criterion_codes"]
            rated_2 = [c for c in codes if self.criterion_has_rating_2(c)]
            met = len(rated_2) >= int(gate["min_rated_2"])
            target = gate["if_met"] if met else gate["if_not_met"]
            record.update({
                "status": "gate",
                "gate_met": met,
                "criteria_rated_2": rated_2,
                "min_rated_2": int(gate["min_rated_2"]),
                "routed_to": target,
                "note": ("Source-defined routing evaluated on PROVISIONAL chatbot ratings "
                        "('?' counts as not '2'); clinician to verify."),
            })
            return record, (None if target == END_MARKER else target)

        if item.get("derived_any_rated_2_of"):
            sources = item["derived_any_rated_2_of"]
            hit = [c for c in sources if self.criterion_has_rating_2(c)]
            record["status"] = "derived"
            record["provisional_rating"] = {
                "value": "2" if hit else "?", "scheme": "scid5pd",
                "label": "provisional – for clinician review",
                "derived_from": sources, "rated_2": hit,
            }
        elif item.get("rating_scheme", "none") != "none":
            record["provisional_rating"] = {"value": "?", "scheme": item["rating_scheme"],
                                            "label": "not assessed by chatbot – clinician rating required"}
        if item.get("same_as"):
            source_id = item["same_as"]
            source_record = next(
                (r for r in self.records if r.get("item_id") == source_id),
                None,
            )

            if source_record and source_record.get("status") == "answered":
                record["status"] = "reused"
                record["same_as"] = source_id
                record["provisional_rating"] = source_record.get("provisional_rating")
                record["evidence_quote"] = source_record.get("evidence_quote")
                record["note"] = "Reused from previously assessed SCID item."

        return record, self._successor_of(item)

    def presentation_text(self, item: dict | None = None) -> str:
        """Text shown when an item is first presented: optional source intro + question."""
        item = item or self.current_question()
        if item is None:
            return ""
        return f"{item['intro']} {item['text']}".strip() if item.get("intro") else item["text"]

    def _reset_item_state(self) -> None:
        """Clear everything that belongs to the current item only."""
        self._awaiting_followup_answer = False
        self._followup_used_for_current = False
        self.pending_followup_text = None
        #EMSAG new parameters were added:
        self._clarifications_used_for_current = 0
        self._pending_followup_kind = None
        #EMSAG new parameters were added:
        # What the participant has said about the CURRENT item, in order,
        # paired with the prompt they were answering. Given to the Judgment
        # Agent so a clarification answer is judged together with the first
        # answer, not in isolation.
        self.item_turns: list[dict] = []
        self.consecutive_system_errors = 0
        # EMSAG: Track unsuccessful attempts separately by response type.
        self._interaction_counts = {
                "redirect": 0,
                "clarify": 0,
                "gentle": 0,
    }

    # ------------------------------------------------------------------
    # Legacy single follow-up (kept for compatibility with older code/tests)
    # ------------------------------------------------------------------
    def start_followup(self, followup_text: str) -> None:
        """Record that a (legacy) follow-up question was asked for the current item."""
        self._awaiting_followup_answer = True
        self._followup_used_for_current = True
        self.pending_followup_text = followup_text
        self._pending_followup_kind = "followup"

    def is_awaiting_followup_answer(self) -> bool:
        """True if the participant's next answer responds to a follow-up/clarification."""
        return self._awaiting_followup_answer

    def followup_already_used(self) -> bool:
        """True if the legacy follow-up has already been used for the current item."""
        return self._followup_used_for_current

    # ------------------------------------------------------------------
    # Clarification budget
    # ------------------------------------------------------------------
    def max_clarifications(self) -> int:
        """Configured clarification budget of the current item (0 if clarification is disabled)."""
        question = self.current_question()
        if question is None or not question.get("allow_clarification", True):
            return 0
        return question.get("max_clarifications", self.default_max_clarifications)

    def current_clarification_count(self) -> int:
        """Number of clarifications/reformulations already used for the current item."""
        return self._clarifications_used_for_current

    def current_interaction_count(self, kind: str) -> int:
        """Return the number of previous follow-ups for this interaction type."""
        return self._interaction_counts.get(kind, 0)

    def clarification_limit_reached(self) -> bool:
        """True if no further clarification is permitted for the current item."""
        return self._clarifications_used_for_current >= self.max_clarifications()

    #EMSAG: RT01: Clarification Request should not cut the budget
    # EMSAG: RT01: Allow unlimited clarification and rephrasing.
    def start_clarification(self, clarification_text: str, kind: str = "clarification") -> bool:
        """
        Store a clarification or rephrasing for the current item.
        Clarification count is tracked for documentation, not as a limit.
        """
        question = self.current_question()

        if question is None:
            return False

        # EMSAG: Respect explicit item-level clarification restrictions.
        if not question.get("allow_clarification", True):
            return False

        # Keep the total count for session records.
        self._clarifications_used_for_current += 1

        # Rephrasing requests do not consume redirect/clarification attempts.
        if kind in self._interaction_counts:
            self._interaction_counts[kind] += 1

        self._awaiting_followup_answer = True
        self._pending_followup_kind = kind
        self.pending_followup_text = clarification_text
        return True

    #EMSAG: RT01: Clarification Request should not cut the budget
    def pending_kind(self) -> str | None:
        """Kind of the pending clarification (public accessor; replaces reading _pending_followup_kind)."""
        return self._pending_followup_kind

    def clear_followup(self) -> None:
        """Clear the pending follow-up/clarification without advancing."""
        self._awaiting_followup_answer = False
        self.pending_followup_text = None
        self._pending_followup_kind = None

    # ------------------------------------------------------------------
    # Per-item evidence and records
    # ------------------------------------------------------------------
    def active_prompt_text(self) -> str:
        """The text the participant is currently answering (clarification if pending, else the item)."""
        question = self.current_question()
        if question is None:
            return ""
        return self.pending_followup_text or self.presentation_text(question)

    def record_participant_turn(self, answer_text: str) -> None:
        """Store one participant message for the current item, with the prompt it answered."""
        self.item_turns.append({"prompt": self.active_prompt_text(), "answer": answer_text})

    def finish_item(self, record: dict, next_item_id: str | None) -> dict | None:
        """
        Close the current item with a record and move to next_item_id.

        The Orchestrator must have validated next_item_id already (it must
        be None for the end, or an existing item id).

        Args:
            record: The item outcome (see session_record.build_item_record()).
            next_item_id: The validated successor.

        Returns:
            The new current item or None at the end.
        """
        self.records.append(record)
        return self._move_to(next_item_id)

    def reset(self) -> None:
        """Start a clean interview using the original guide configuration."""

        # EMSAG: 02.11.2026
        # Restore the original question configuration before starting a new
        # participant session. This prevents temporary SZD_A5 changes from
        # carrying over to subsequent participants.
        self.questions = copy.deepcopy(self._original_questions)

        self._current_index = 0
        self.records: list[dict] = []
        self.paused = False
        self._reset_item_state()
        self._skip_non_interactive()