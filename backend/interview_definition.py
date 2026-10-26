"""
INTERVIEW DEFINITION — V2 LOADER, IMPORT-TIME VALIDATION, LEGACY ADAPTER
=======================================================================
The InterviewDefinition loader for the v2 format — the "Interview Navi"
design (docs/interview_navi_design.md is the binding reference):

  * the hierarchical structure  Section Group -> Question Group ->
    Single Question; section groups nest arbitrarily deep (section in
    section) and may declare configurable relations to other sections
    (temporal / pathological / custom),
  * the type-encoding id scheme  sec_* / qg_* / sq_*  (pragmatic rule:
    the prefix is mandatory and validated against the element's type;
    the numeric position part is INFORMATIONAL — assigned at creation,
    never renumbered or re-validated on edits, so ids stay stable),
  * per-question answer types (boolean = yes/no, number, string),
  * question groups with a traversal strategy (all_required /
    sequential today; the foreseen dependent / complex strategies come
    with the Navigator) and a criteria-group marker,
  * criteria: declared ON a criteria group (question group OR section
    group); a criteria group's criterion always refers to ALL elements
    of that group together. Rule AST: all / any / not / eq / gte /
    lte / threshold (the design note's full operator set — logical,
    arithmetic, text, completeness, group-level aggregation — is a
    later step),
  * transitions (jumps) as direct id references to any single question,
    question group or section group. Jumps may form LOOPS (the earlier
    loop-free requirement is dropped); only dead ends — missing or
    invalid jump targets — are rejected at import,
  * declared navigation values, required meaning units per question,
    the behavior_profile link.

What this module deliberately does NOT do:
  * it does not evaluate criteria
  * it does not pick the next question
  * it does not run the interview

Format detection (load_interview_definition): a YAML file with a
top-level "definition_version" key is v2 (v2 definitions always pin
their version; legacy guides never have one); anything else is the
legacy linear format and goes through the adapter, so existing guide
files keep loading with the same question order and behavior.

Import-time validation (ALL problems collected in one pass, then raised
together as InterviewDefinitionError with an actionable message):
  * unique ids: sections, question groups, questions, criteria (across
    all groups), value names, meaning units
  * the id prefix matches the element's type (sec_ section, qg_
    question group, sq_ question)
  * jump targets resolve to an existing section / question group /
    question id (dead ends are detected and prevented at import)
  * jump conditions (on_criterion) reference a declared criterion
  * section relations point at another existing section group
  * a criteria-group marker and the group's criteria go together
  * criteria reference declared values only, with matching types:
    eq checks the literal against the declared type; gte/lte/threshold
    need a number value; threshold names must be declared on the value
  * category values list at least one (unique) category; number values
    respect min <= max
  * language is one of the supported languages (de / en, see
    study_content/study_settings.yaml)
  * behavior_profile (when set) exists in study_content/prompts/profiles.yaml
  * v2 files contain at least one question (legacy files keep their
    legacy behavior — an empty legacy guide still loads)

Jumps may loop; whether a loop may re-enter an already locked group is
an open point (RECONSTRUCTION_PLAN.md) decided when the Navigator
(step 2.5) is built.

Legacy id compatibility: the adapter maps each legacy section onto one
section group and GENERATES v2-convention ids (sec_<n>, qg_<n>.1,
sq_<n>.1.<m>); legacy question ids (e.g. "dep_01") are remapped, but
the question ORDER and all other behavior stay identical.

Used by: backend/interview_guide.py (legacy pipeline via the adapter),
backend/interview_run.py (the run state machine) and the Supervisor /
pipeline components.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal, Mapping, Sequence, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from behavior_profiles import PROFILES_PATH
from contracts import CriterionStatus

# All interview guide files live in this folder (researcher-editable area).
INTERVIEWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "study_content",
    "interviews",
)

# Languages supported end to end today (STT language + TTS voices).
SUPPORTED_LANGUAGES = frozenset({"de", "en"})

# Id scheme (pragmatic rule): the PREFIX is mandatory and validated
# against the element's type; the numeric position part after the
# prefix (e.g. "1", "1.1", "1.1.1") is INFORMATIONAL — assigned at
# creation, never renumbered or re-validated on edits, so ids stay
# stable across edits (they also serve as stable longitudinal
# identifiers).
SECTION_ID_PREFIX = "sec_"
QUESTION_GROUP_ID_PREFIX = "qg_"
QUESTION_ID_PREFIX = "sq_"


class InterviewDefinitionError(ValueError):
    """
    The interview definition file is malformed or fails the import-time
    validation. The message lists ALL problems found in one pass so the
    researcher can fix them in one round trip.
    """


# ---------------------------------------------------------------------------
# v2 definition models (frozen; a change produces a new model — same
# convention as backend/contracts.py)
# ---------------------------------------------------------------------------


class MeaningUnit(BaseModel):
    """
    One required semantic unit of a question's answer — the "required
    meaning units" of the plan (docs/interview_navi_design.md has no
    counterpart; the concept is orthogonal to it): the content the
    answer must cover for the question to count as answered (id + short
    description).

    How it is used: it is the extraction contract for the Diagnostiker
    (step 2.3 / Phase 3) — an answer covering all of the question's
    meaning units can be CLOSED; an answer that misses a unit drives the
    UNCLEAR/REVISED clarification cycle (see contracts.QuestionState).
    The expert reference scenarios (Phase 7) use the same units as the
    expected answer content. Until the evaluator exists, meaning units
    are declarative only: this module validates them (globally unique
    ids) but nothing here consumes them yet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Jump(BaseModel):
    """
    One declarative transition (jump) out of a question.

    `to` names a single question, a question group (the jump lands on
    its first question) or a section group (the jump lands on the
    section's entry point — the first question in postorder traversal).
    `on_criterion` (optional) gates the jump on a criterion: the jump
    is only taken when that criterion reaches `when` (default
    SATISFIED). Without `on_criterion` the jump is unconditional.

    Jumps are data here — they are evaluated by the Navigator, not by
    this module. Jumps MAY form loops (design note); only dead ends
    (unknown target) are rejected at import.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    to: str = Field(min_length=1)
    on_criterion: str | None = None
    when: CriterionStatus = CriterionStatus.SATISFIED


# How the answer to a single question is recorded (design note:
# Ja/Nein, Zahl, unspezifizierter String).
AnswerType = Literal["boolean", "number", "string"]


class QuestionDefinition(BaseModel):
    """
    One single question of the v2 format. `text` is the canonical
    semantic wording (verbatim data, never generated); the Sprecher
    may rephrase the spoken formulation by default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    # How the answer is recorded: boolean (yes/no), number (a scale
    # value) or string (free-form — the default, matching the legacy
    # "just ask and record" behavior).
    answer_type: AnswerType = "string"
    # all_required groups require every required question to be asked
    # before the group is done; the Navigator may skip required=False.
    required: bool = True
    # Legacy-compat flag (the old pipeline allowed one short follow-up).
    # In the new protocol, clarification is driven by the UNCLEAR criterion
    # cycle instead; the flag is kept so legacy guides map losslessly.
    allow_followup: bool = False
    meaning_units: list[MeaningUnit] = Field(default_factory=list)
    jumps: list[Jump] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_a_question_id(cls, v: str) -> str:
        if not v.startswith(QUESTION_ID_PREFIX):
            raise ValueError(
                f"question id {v!r} must start with {QUESTION_ID_PREFIX!r} "
                f"(id scheme sec_/qg_/sq_)"
            )
        return v


# How a question group is traversed (the Navigator reads this):
#   sequential   — questions are asked in list order.
#   all_required — every required question of the group must be asked
#                  (order is the Navigator's choice). This is the safe
#                  default.
# The design note foresees further strategies (dependent / complex);
# they are added when the Navigator is built (step 2.5).
GroupStrategy = Literal["all_required", "sequential"]


class QuestionGroup(BaseModel):
    """
    One question group: a thematic bundle of single questions with a
    configurable internal traversal strategy.

    A question group MAY be marked as a criteria group
    (criteria_group=True): its criteria then always refer to ALL
    elements of the group together (the group-level scope of the design
    note). Marker and criteria must go together (cross-validated).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    strategy: GroupStrategy = "all_required"
    # A question group without questions is dead weight — rejected.
    questions: list[QuestionDefinition] = Field(
        default_factory=list, min_length=1
    )
    # Criteria-group marker (see the class docstring).
    criteria_group: bool = False
    criteria: list[CriterionDefinition] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_a_question_group_id(cls, v: str) -> str:
        if not v.startswith(QUESTION_GROUP_ID_PREFIX):
            raise ValueError(
                f"question group id {v!r} must start with "
                f"{QUESTION_GROUP_ID_PREFIX!r} (id scheme sec_/qg_/sq_)"
            )
        return v


class ValueDeclaration(BaseModel):
    """
    One declared navigation value: a NAMED, TYPED input the criterion
    rule engine reads — the "navigation-relevant values" the Diagnostiker
    extracts per answer (plan §1.1). Criteria do not parse raw answer
    text: they reference these values by name, and the rule AST
    (eq / gte / lte / threshold / all / any / not) is evaluated over
    them. Import-time validation guarantees a criterion only references
    declared values, with type-matching operators.

    Relation to the design note: the note's per-question answer type
    (boolean / number / string) defines how a question's answer is
    RECORDED; values are the separate, definition-level indirection that
    feeds the CRITERIA — a criteria group's rule may combine values set
    from several of the group's questions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    type: Literal["bool", "number", "category"]
    # type=category: the allowed categories (must be non-empty, unique).
    categories: list[str] = Field(default_factory=list)
    # type=number: optional scale bounds and NAMED thresholds
    # (e.g. thresholds: {mild: 1, moderate: 5}) that the `threshold`
    # rule operator can refer to.
    min: float | None = None
    max: float | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)


# --- the declarative criterion AST (the rule operators of the plan: -----
# --- all / any / not / eq / gte / lte / threshold) -----------------------


class EqRule(BaseModel):
    """`value == equals`, checked against the value's declared type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["eq"] = "eq"
    value: str = Field(min_length=1)
    equals: bool | int | float | str


class GteRule(BaseModel):
    """`value >= threshold` (number values only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["gte"] = "gte"
    value: str = Field(min_length=1)
    threshold: float


class LteRule(BaseModel):
    """`value <= threshold` (number values only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["lte"] = "lte"
    value: str = Field(min_length=1)
    threshold: float


class ThresholdRule(BaseModel):
    """
    `value >= <named threshold>` where the threshold value is declared on
    the value itself (researcher-editable in the YAML, e.g.
    thresholds: {moderate: 5}) — not a magic number inside the rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["threshold"] = "threshold"
    value: str = Field(min_length=1)
    threshold: str = Field(min_length=1)


class AllRule(BaseModel):
    """All sub-rules hold (at least one rule)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["all"] = "all"
    rules: list["CriteriaRule"] = Field(min_length=1)


class AnyRule(BaseModel):
    """At least one sub-rule holds (at least one rule)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["any"] = "any"
    rules: list["CriteriaRule"] = Field(min_length=1)


class NotRule(BaseModel):
    """The sub-rule does not hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["not"] = "not"
    rule: "CriteriaRule"


CriteriaRule = Annotated[
    Union[EqRule, GteRule, LteRule, ThresholdRule, AllRule, AnyRule, NotRule],
    Field(discriminator="op"),
]


class CriterionDefinition(BaseModel):
    """
    One declared criterion: a declarative rule over navigation values.
    Evaluation (the four-valued result + "decided" computation)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str | None = None
    rule: CriteriaRule


class SectionRelation(BaseModel):
    """
    One configurable relation from this section group to ANOTHER section
    group (design note: relations between sections are freely definable):
    temporal (course of time), pathological (clinical relevance between
    the two sections' content) or custom (free).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    to: str = Field(min_length=1)
    kind: Literal["temporal", "pathological", "custom"]
    note: str = ""


class SectionGroup(BaseModel):
    """
    One section group: a thematic container for (nested) section groups
    and question groups. Section groups nest arbitrarily deep, including
    section in section.

    A section group MAY be marked as a criteria group (criteria_group=
    True): its criteria then always refer to ALL elements of the whole
    section together (design note). Marker and criteria must go together
    (cross-validated).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    # Nested section groups and question groups, in document order
    # (the Navigator's default postorder traversal follows this order).
    children: list["SectionOrGroup"] = Field(default_factory=list)
    # Configurable relations to other section groups.
    relations: list[SectionRelation] = Field(default_factory=list)
    # Criteria-group marker (see the class docstring).
    criteria_group: bool = False
    criteria: list[CriterionDefinition] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_must_be_a_section_id(cls, v: str) -> str:
        if not v.startswith(SECTION_ID_PREFIX):
            raise ValueError(
                f"section id {v!r} must start with {SECTION_ID_PREFIX!r} "
                f"(id scheme sec_/qg_/sq_)"
            )
        return v


# A node of the interview tree: a section group or a question group.
# The two are distinguished by the id prefix (validated by the models'
# own id validators), so a plain union resolves unambiguously.
SectionOrGroup = SectionGroup | QuestionGroup

# Resolve the forward references created by `from __future__ import
# annotations` (SectionGroup.children -> SectionOrGroup; the criteria
# lists -> CriterionDefinition, defined above).
SectionGroup.model_rebuild()
QuestionGroup.model_rebuild()


class InterviewDefinition(BaseModel):
    """
    The v2 interview definition: the authoritative structure the
    Navigator operates on — a tree of section groups, question groups
    and single questions (see docs/interview_navi_design.md).

    Cross-field validation (unique ids, id prefixes, jump targets,
    criteria/value consistency) runs in _cross_validate on every
    construction — the "import-time validation" of the plan. Jumps may
    form loops (allowed by design); only dead ends are rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Pinned version of this definition; every run state and every audit
    # event carries it (contracts.RunState.definition_version). Its
    # presence at the top level of a YAML file also marks the file as
    # v2 (format detection in load_interview_definition).
    definition_version: str = Field(min_length=1)
    guide_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    language: str = "de"
    # Link to study_content/prompts/profiles.yaml (the Sprecher's
    # configuration input and the Validator's constraint).
    behavior_profile: str | None = None
    framing_message: str = ""
    closing_message: str = ""
    # Pre-approved fallback formulation for the Validator's canonical
    # bypass; falls back to the default in contracts when left empty.
    canonical_fallback_message: str | None = None
    values: list[ValueDeclaration] = Field(default_factory=list)
    # The hierarchy: top-level section groups. Criteria live on the
    # (question/section) groups themselves — a criteria group's
    # criterion refers to all elements of that group together.
    sections: list[SectionGroup] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # derived views — document order (depth-first, children in listed
    # order). Postorder NAVIGATION is the Navigator's job (step 2.5),
    # not this module's; these accessors are plain data views.
    # ------------------------------------------------------------------

    def _walk(self) -> Iterator[tuple[str, SectionOrGroup]]:
        """Yield (path, node) for every section/question group, document order."""

        def walk(
            nodes: Sequence[SectionOrGroup], trail: list[str]
        ) -> Iterator[tuple[str, SectionOrGroup]]:
            for node in nodes:
                if isinstance(node, SectionGroup):
                    yield " > ".join(trail + [node.id]), node
                    yield from walk(node.children, trail + [node.id])
                else:
                    yield " > ".join(trail + [node.id]), node

        yield from walk(self.sections, [])

    def all_section_groups(self) -> list[SectionGroup]:
        """All section groups (every depth), in document order."""
        return [node for _, node in self._walk() if isinstance(node, SectionGroup)]

    def all_question_groups(self) -> list[QuestionGroup]:
        """All question groups (every depth), in document order."""
        return [node for _, node in self._walk() if isinstance(node, QuestionGroup)]

    def all_questions(self) -> list[QuestionDefinition]:
        """All single questions, in document order (their groups' order)."""
        return [
            q
            for _, node in self._walk()
            if isinstance(node, QuestionGroup)
            for q in node.questions
        ]

    def questions_with_group(self) -> list[tuple[QuestionGroup, QuestionDefinition]]:
        """All single questions with their question group, in document order."""
        return [
            (node, q)
            for _, node in self._walk()
            if isinstance(node, QuestionGroup)
            for q in node.questions
        ]

    def all_criteria(self) -> list[CriterionDefinition]:
        """All criteria (of all criteria groups), in document order."""
        return [c for _, node in self._walk() for c in node.criteria]

    def question_by_id(self, question_id: str) -> QuestionDefinition | None:
        """The single question with this id, or None."""
        return next((q for q in self.all_questions() if q.id == question_id), None)

    def group_by_id(self, group_id: str) -> QuestionGroup | None:
        """The question group with this id, or None."""
        return next((g for g in self.all_question_groups() if g.id == group_id), None)

    def section_by_id(self, section_id: str) -> SectionGroup | None:
        """The section group with this id, or None."""
        return next((s for s in self.all_section_groups() if s.id == section_id), None)

    def criterion_by_id(self, criterion_id: str) -> CriterionDefinition | None:
        """The criterion with this id (of any criteria group), or None."""
        return next((c for c in self.all_criteria() if c.id == criterion_id), None)

    @model_validator(mode="after")
    def _cross_validate(self) -> InterviewDefinition:
        errors: list[str] = []

        # --- declared-value lookups used by the rule checks -------------
        values = {v.name: v for v in self.values}

        # --- 1. unique ids + node inventory (document order) -----------
        # (sections / question groups / questions / criteria / values /
        # meaning units — all ids are globally unique)
        seen: dict[str, str] = {}

        def note(kind: str, item_id: str, location: str) -> None:
            if item_id in seen:
                errors.append(
                    f"duplicate {kind} id {item_id!r} "
                    f"(first at {seen[item_id]}, again at {location})"
                )
            else:
                seen[item_id] = location

        section_ids: set[str] = set()
        group_ids: set[str] = set()
        question_ids: set[str] = set()
        criterion_ids: set[str] = set()

        for path, node in self._walk():
            if isinstance(node, SectionGroup):
                note("section", node.id, f"section {node.id!r} ({path})")
                section_ids.add(node.id)
            else:
                note("group", node.id, f"question group {node.id!r} ({path})")
                group_ids.add(node.id)
            for ci, c in enumerate(node.criteria):
                note(
                    "criterion",
                    c.id,
                    f"criterion {c.id!r} of {path} (criteria[{ci}])",
                )
                criterion_ids.add(c.id)
            if isinstance(node, QuestionGroup):
                for qi, q in enumerate(node.questions):
                    note(
                        "question",
                        q.id,
                        f"question {q.id!r} in {path} (questions[{qi}])",
                    )
                    question_ids.add(q.id)
                    for ui, unit in enumerate(q.meaning_units):
                        note(
                            "meaning-unit",
                            unit.id,
                            f"meaning unit {unit.id!r} of {q.id!r} in {path} "
                            f"(meaning_units[{ui}])",
                        )
        for vi, v in enumerate(self.values):
            note("value", v.name, f"value {v.name!r} (values[{vi}])")

        # --- 2. criteria-group marker and criteria go together ----------
        for path, node in self._walk():
            if node.criteria_group and not node.criteria:
                errors.append(
                    f"{path}: marked as a criteria group but has no criteria"
                )
            elif node.criteria and not node.criteria_group:
                errors.append(
                    f"{path}: has criteria but is not marked as a criteria "
                    f"group (criteria_group: true)"
                )

        # --- 3. section relations point at another existing section -----
        for path, node in self._walk():
            if not isinstance(node, SectionGroup):
                continue
            for ri, rel in enumerate(node.relations):
                rloc = f"{path}, relations[{ri}]"
                if rel.to == node.id:
                    errors.append(f"{rloc}: a section cannot relate to itself")
                elif rel.to not in section_ids:
                    errors.append(
                        f"{rloc}: relation target {rel.to!r} is not a known "
                        f"section id (sections: {sorted(section_ids) or 'none'})"
                    )

        # --- 4. jump targets + jump conditions (dead ends) --------------
        # No acyclicity check on purpose: jumps MAY form loops (design
        # note). The loop-back-to-locked-group behavior is an open point
        # decided with the Navigator (step 2.5).
        for path, node in self._walk():
            if not isinstance(node, QuestionGroup):
                continue
            for qi, q in enumerate(node.questions):
                qloc = f"question {q.id!r} in {path} (questions[{qi}])"
                for ji, jump in enumerate(q.jumps):
                    jloc = f"{qloc}, jumps[{ji}]"
                    if (
                        jump.to not in section_ids
                        and jump.to not in group_ids
                        and jump.to not in question_ids
                    ):
                        errors.append(
                            f"{jloc}: jump target {jump.to!r} is not a known "
                            f"section, group or question id (sections: "
                            f"{sorted(section_ids) or 'none'}, groups: "
                            f"{sorted(group_ids) or 'none'}, questions: "
                            f"{sorted(question_ids) or 'none'})"
                        )
                    if jump.on_criterion is not None and (
                        jump.on_criterion not in criterion_ids
                    ):
                        errors.append(
                            f"{jloc}: on_criterion {jump.on_criterion!r} is not "
                            f"a declared criterion (criteria: "
                            f"{sorted(criterion_ids) or 'none'})"
                        )

        # --- 4. criteria reference declared values with matching types ---
        def value_decl(name: str, path: str) -> ValueDeclaration | None:
            decl = values.get(name)
            if decl is None:
                errors.append(
                    f"{path}: value {name!r} is not declared in 'values' "
                    f"(declared: {sorted(values) or 'none'})"
                )
            return decl

        def check_rule(rule: CriteriaRule, path: str) -> None:
            if isinstance(rule, (AllRule, AnyRule)):
                for i, sub in enumerate(rule.rules):
                    check_rule(sub, f"{path}.rules[{i}]")
            elif isinstance(rule, NotRule):
                check_rule(rule.rule, f"{path}.rule")
            elif isinstance(rule, EqRule):
                decl = value_decl(rule.value, path)
                if decl is None:
                    return
                ok = (
                    decl.type == "bool" and isinstance(rule.equals, bool)
                ) or (
                    decl.type == "number"
                    and isinstance(rule.equals, (int, float))
                    and not isinstance(rule.equals, bool)
                ) or (
                    decl.type == "category"
                    and isinstance(rule.equals, str)
                    and rule.equals in decl.categories
                )
                if not ok:
                    details = f"(type: {decl.type}"
                    if decl.categories:
                        details += f", categories: {decl.categories}"
                    details += ")"
                    errors.append(
                        f"{path}: eq {rule.equals!r} does not match the "
                        f"declared type of value {rule.value!r} {details}"
                    )
            elif isinstance(rule, (GteRule, LteRule)):
                decl = value_decl(rule.value, path)
                if decl is not None and decl.type != "number":
                    errors.append(
                        f"{path}: {rule.op} needs a number value, but "
                        f"{rule.value!r} is declared as {decl.type}"
                    )
            elif isinstance(rule, ThresholdRule):
                decl = value_decl(rule.value, path)
                if decl is None:
                    return
                if decl.type != "number":
                    errors.append(
                        f"{path}: threshold needs a number value, but "
                        f"{rule.value!r} is declared as {decl.type}"
                    )
                elif rule.threshold not in decl.thresholds:
                    errors.append(
                        f"{path}: threshold {rule.threshold!r} is not declared "
                        f"on value {rule.value!r} (declared thresholds: "
                        f"{sorted(decl.thresholds) or 'none'})"
                    )

        for path, node in self._walk():
            for ci, c in enumerate(node.criteria):
                check_rule(c.rule, f"criterion {c.id!r} of {path} (criteria[{ci}])")

        # --- 5. value declarations are well-formed ------------------------
        for vi, v in enumerate(self.values):
            vloc = f"value {v.name!r} (values[{vi}])"
            if v.type == "category" and not v.categories:
                errors.append(
                    f"{vloc}: category values need at least one 'categories' entry"
                )
            if v.type == "category" and len(set(v.categories)) != len(v.categories):
                errors.append(f"{vloc}: duplicate entries in 'categories'")
            if (
                v.type == "number"
                and v.min is not None
                and v.max is not None
                and v.min > v.max
            ):
                errors.append(f"{vloc}: min ({v.min}) must not exceed max ({v.max})")
            if v.type == "bool" and (
                v.categories or v.thresholds or v.min is not None or v.max is not None
            ):
                errors.append(
                    f"{vloc}: bool values cannot carry categories/thresholds/min/max"
                )

        # --- 6. language ---------------------------------------------------
        if self.language not in SUPPORTED_LANGUAGES:
            errors.append(
                f"language {self.language!r} is not supported "
                f"(supported: {sorted(SUPPORTED_LANGUAGES)})"
            )

        if errors:
            raise ValueError(
                "invalid interview definition:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return self


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _known_profile_names() -> set[str]:
    """The profile names defined in study_content/prompts/profiles.yaml."""
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}
    if not isinstance(registry, dict):
        raise InterviewDefinitionError(
            f"{PROFILES_PATH}: top level must be a YAML mapping of "
            f"profile name -> profile entry"
        )
    return {str(name) for name in registry}


def validate_definition(
    definition: InterviewDefinition,
    source_name: str,
    *,
    require_questions: bool = True,
) -> InterviewDefinition:
    """
    The loader-level checks that need file access or a format decision:
    behavior_profile must be a known profile, and v2 files must contain at
    least one question (legacy files keep their legacy behavior, where an
    empty guide is loadable). Returns the definition, or raises
    InterviewDefinitionError with all problems.
    """
    errors: list[str] = []
    if definition.behavior_profile is not None:
        known = _known_profile_names()
        if definition.behavior_profile not in known:
            errors.append(
                f"behavior_profile {definition.behavior_profile!r} is not "
                f"defined in study_content/prompts/profiles.yaml "
                f"(known: {sorted(known) or 'none'})"
            )
    if require_questions and not definition.all_questions():
        errors.append(
            "definition has no questions (a v2 guide needs at least one question)"
        )
    if errors:
        raise InterviewDefinitionError(
            f"invalid interview definition {source_name!r}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    return definition


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise InterviewDefinitionError(
            f"{path}: top level of an interview definition must be a YAML mapping"
        )
    return data


def _parse_v2(data: dict[str, Any], source: str) -> InterviewDefinition:
    try:
        definition = InterviewDefinition.model_validate(data)
    except ValidationError as exc:
        raise InterviewDefinitionError(
            f"invalid v2 interview definition {source!r}:\n{exc}"
        ) from exc
    return validate_definition(definition, source, require_questions=True)


def load_v2_definition(path: str | Path) -> InterviewDefinition:
    """
    Load one v2 interview definition file (strict validation, including:
    at least one question).
    """
    return _parse_v2(_load_yaml_mapping(path), os.path.basename(str(path)))


def load_legacy_guide(path: str | Path) -> InterviewDefinition:
    """
    Load one LEGACY linear guide file through the v2 adapter (same
    question order and behavior as the old InterviewGuide).
    """
    source = os.path.basename(str(path))
    return validate_definition(
        to_v2(_load_yaml_mapping(path), source), source, require_questions=False
    )


def load_interview_definition(path: str | Path) -> InterviewDefinition:
    """
    Load an interview definition file, auto-detecting the format:
    a top-level "definition_version" key means v2 (v2 definitions always
    pin their version; legacy guides never have one), anything else is
    the legacy linear format (mapped through the adapter). This is the
    single entry point the new pipeline (Phase 2, step 2.5 onward) uses.
    """
    data = _load_yaml_mapping(path)
    source = os.path.basename(str(path))
    if "definition_version" in data:
        return _parse_v2(data, source)
    return validate_definition(to_v2(data, source), source, require_questions=False)


# ---------------------------------------------------------------------------
# legacy adapter (legacy linear YAML -> v2 model)
# ---------------------------------------------------------------------------


def to_v2(legacy_data: Mapping[str, Any], guide_filename: str = "") -> InterviewDefinition:
    """
    Map one legacy guide (sections -> one section group each, one
    question group per section, strictly sequential, no
    values/criteria/jumps) onto the v2 model.

    The legacy guide keeps its question order and behavior:
      * each section becomes one section group with a single
        sequential question group
      * every legacy question is required (the legacy flow asks all of
        them)
      * allow_followup is carried over (legacy-compat flag)
      * ids are GENERATED with the v2 convention (sec_<n>, qg_<n>.1,
        sq_<n>.1.<m> — positional at creation time, informational
        thereafter); legacy question ids (e.g. "dep_01") are remapped
      * guide_id falls back to the file name, exactly like the old loader
      * definition_version is DERIVED: "<guide_id>@legacy-<8 hex chars>"
        where the hash is over the guide's content (guide id, messages,
        question list) — stable across loads of the same file, and it
        changes when the guide changes (so audit events stay attributable)
    """
    if not isinstance(legacy_data, Mapping):
        raise InterviewDefinitionError(
            "legacy guide data must be a mapping (top-level YAML mapping)"
        )

    guide_id = str(legacy_data.get("guide_id") or guide_filename or "legacy_guide")
    title = str(legacy_data.get("title") or guide_id)
    framing = str(legacy_data.get("framing_message") or "").strip()
    closing = str(legacy_data.get("closing_message") or "").strip()
    raw_profile = legacy_data.get("behavior_profile")
    behavior_profile = None if raw_profile is None else str(raw_profile)

    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    sections = legacy_data.get("sections") or []
    if not isinstance(sections, list):
        sections = []
        errors.append("'sections' must be a list (of section mappings)")
    for si, section in enumerate(sections):
        if not isinstance(section, Mapping):
            errors.append(f"sections[{si}] must be a mapping")
            continue
        section_title = section.get("section_title")
        questions = section.get("questions") or []
        if not isinstance(questions, list):
            errors.append(f"sections[{si}].questions must be a list")
            continue
        for qi, q in enumerate(questions):
            if not isinstance(q, Mapping):
                errors.append(f"sections[{si}].questions[{qi}] must be a mapping")
                continue
            qid = q.get("id")
            text = q.get("text")
            if not qid:
                errors.append(f"sections[{si}].questions[{qi}] has no 'id'")
            if not text:
                errors.append(
                    f"sections[{si}].questions[{qi}] (id: {qid!r}) has no 'text'"
                )
            entries.append(
                {
                    "section_index": si,
                    "section_title": section_title,
                    "id": str(qid),
                    "text": str(text),
                    "allow_followup": bool(q.get("allow_followup", False)),
                }
            )
    if errors:
        raise InterviewDefinitionError(
            f"invalid legacy guide {guide_filename or '(data)'}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # Map the flat entries onto the hierarchy (stable order): one
    # section group per legacy section, one question group per section.
    sections_v2: list[SectionGroup] = []
    for si in range(len(sections)):
        section_entries = [e for e in entries if e["section_index"] == si]
        if not section_entries:
            continue  # a section without (valid) questions contributes nothing
        section_title = next(
            (e["section_title"] for e in section_entries if e["section_title"]),
            None,
        )
        section_label = (
            str(section_title) if section_title else f"Section {si + 1}"
        )
        sections_v2.append(
            SectionGroup(
                id=f"sec_{si + 1}",
                title=section_label,
                children=[
                    QuestionGroup(
                        id=f"qg_{si + 1}.1",
                        title=section_label,
                        strategy="sequential",
                        questions=[
                            QuestionDefinition(
                                id=f"sq_{si + 1}.1.{qi + 1}",
                                text=e["text"],
                                required=True,
                                allow_followup=e["allow_followup"],
                            )
                            for qi, e in enumerate(section_entries)
                        ],
                    )
                ],
            )
        )

    canonical = json.dumps(
        {
            "guide_id": guide_id,
            "behavior_profile": behavior_profile,
            "framing_message": framing,
            "closing_message": closing,
            "questions": [
                {
                    "section_index": e["section_index"],
                    "id": e["id"],
                    "text": e["text"],
                    "allow_followup": e["allow_followup"],
                }
                for e in entries
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    return InterviewDefinition(
        definition_version=f"{guide_id}@legacy-{digest}",
        guide_id=guide_id,
        title=title,
        behavior_profile=behavior_profile,
        framing_message=framing,
        closing_message=closing,
        sections=sections_v2,
    )
