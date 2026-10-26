"""
Unit tests for backend/interview_definition.py (RECONSTRUCTION_PLAN.md
Phase 2, step 2.1): the v2 hierarchical InterviewDefinition loader
(Section Groups > Question Groups > Single Questions, sections nest
arbitrarily deep), the import-time validation (globally unique ids, id
prefixes sec_/qg_/sq_, section relations, criteria-group marker/criteria
consistency, jump targets, criteria referencing known values/types,
known behavior profiles) and the legacy adapter that maps the current
linear guide YAML onto the v2 model with generated ids.

Loops between jumps are allowed by design (the loop-back-to-locked-group
behavior is an open point for the Navigator, step 2.5); only dead ends
(unknown jump targets) and invalid references are rejected at import.

All tests are deterministic and need no network access.
"""

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pydantic import ValidationError

from contracts import CriterionStatus
from interview_definition import (
    AnyRule,
    CriterionDefinition,
    EqRule,
    GteRule,
    InterviewDefinition,
    InterviewDefinitionError,
    NotRule,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ThresholdRule,
    ValueDeclaration,
    load_interview_definition,
    load_v2_definition,
    to_v2,
)
from interview_guide import InterviewGuide

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERVIEWS_DIR = os.path.join(REPO_ROOT, "study_content", "interviews")

# A valid v2 definition exercising every format feature: a nested section
# hierarchy (section in section), both group strategies, criteria on
# criteria groups (incl. a composite rule AST), meaning units, answer
# types, a section relation and jumps to a question (a LOOP back into the
# same group), to a question group and to a section group.
VALID_V2_YAML = """
definition_version: "test_v2@1"
guide_id: test_v2
title: "V2 Test Guide"
language: de
behavior_profile: strict_clinical
framing_message: "Hi."
closing_message: "Bye."
canonical_fallback_message: "Please answer in one sentence."
values:
  - name: yes_no
    type: bool
  - name: score
    type: number
    min: 0
    max: 10
    thresholds:
      low: 3
      high: 8
  - name: pattern
    type: category
    categories: [a, b]
sections:
  - id: sec_1
    title: "Mood"
    children:
      - id: qg_1.1
        title: "Core"
        strategy: all_required
        criteria_group: true
        criteria:
          - id: c_yes
            rule: {op: eq, value: yes_no, equals: true}
          - id: c_high
            rule: {op: gte, value: score, threshold: 8}
          - id: c_low
            rule: {op: lte, value: score, threshold: 3}
          - id: c_at_low
            rule: {op: threshold, value: score, threshold: low}
        questions:
          - id: sq_1.1.1
            text: "Q one?"
            required: true
            meaning_units:
              - id: mu_one
                description: "unit one"
          - id: sq_1.1.2
            text: "Q two?"
            answer_type: boolean
            jumps:
              # a loop: back to the first question of the same group
              - to: sq_1.1.1
                on_criterion: c_yes
                when: SATISFIED
              # a jump onto a question group (lands on its first question)
              - to: qg_1.1.1
                on_criterion: c_high
                when: NOT_SATISFIED
      - id: sec_1.1
        title: "Sleep (nested section)"
        relations:
          - to: sec_1
            kind: temporal
            note: "nested section relates to its parent"
        children:
          - id: qg_1.1.1
            title: "Course"
            strategy: sequential
            criteria_group: true
            criteria:
              - id: c_mixed
                rule:
                  op: all
                  rules:
                    - op: any
                      rules:
                        - op: eq
                          value: pattern
                          equals: a
                        - op: eq
                          value: yes_no
                          equals: false
                    - op: not
                      rule: {op: lte, value: score, threshold: 0}
            questions:
              - id: sq_1.1.1.1
                text: "Q three?"
                jumps:
                  # a jump onto a section group (unconditional)
                  - to: sec_1
              - id: sq_1.1.1.2
                text: "Q four?"
                answer_type: number
                required: false
"""

LEGACY_YAML = """
guide_id: legacy_test
title: "Legacy Test"
behavior_profile: strict_clinical
framing_message: "  Welcome.  "
closing_message: "Bye."
sections:
  - section_title: "Section A"
    questions:
      - id: q1
        text: "Question one?"
      - id: q2
        text: "Question two?"
        allow_followup: true
  - section_title: "Section B"
    questions:
      - id: q3
        text: "Question three?"
"""



class DefinitionTestBase(unittest.TestCase):
    """Shared helpers: temp YAML files in the interviews dir + loaders."""

    def _write(self, yaml_text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml", dir=INTERVIEWS_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        self.addCleanup(os.remove, path)
        return path

    def _load_v2(self, yaml_text: str) -> InterviewDefinition:
        return load_v2_definition(self._write(yaml_text))

    def _expect_error(self, yaml_text: str, *needles: str) -> str:
        """Load a (bad) v2 file and assert the error message contains all needles."""
        with self.assertRaises(InterviewDefinitionError) as ctx:
            self._load_v2(yaml_text)
        message = str(ctx.exception)
        for needle in needles:
            self.assertIn(needle, message)
        return message

    def _defn_with_rules(self, values: list, rules: list) -> str:
        """
        Build a definition with given value declarations + criterion rules
        directly (bypassing the YAML file) to unit-test the rule checks.
        Returns the (expected) validation error message.
        """
        with self.assertRaises(ValidationError) as ctx:
            InterviewDefinition(
                definition_version="v",
                guide_id="g",
                title="t",
                values=[ValueDeclaration(**v) for v in values],
                sections=[
                    SectionGroup(
                        id="sec_1",
                        title="T",
                        children=[
                            QuestionGroup(
                                id="qg_1.1",
                                title="T",
                                criteria_group=True,
                                questions=[QuestionDefinition(id="sq_1.1.1", text="x")],
                                criteria=[
                                    CriterionDefinition(id=f"c{i}", rule=r)
                                    for i, r in enumerate(rules)
                                ],
                            )
                        ],
                    )
                ],
            )
        return str(ctx.exception)

    def _expect_rule_error(self, values: list, rules: list, *needles: str) -> str:
        message = self._defn_with_rules(values, rules)
        for needle in needles:
            self.assertIn(needle, message)
        return message


class TestV2Loading(DefinitionTestBase):
    """The v2 hierarchical format loads; auto-detection picks the right loader."""

    def test_valid_v2_definition_loads(self):
        defn = self._load_v2(VALID_V2_YAML)
        self.assertEqual(defn.definition_version, "test_v2@1")
        self.assertEqual(defn.guide_id, "test_v2")
        self.assertEqual(defn.behavior_profile, "strict_clinical")
        self.assertEqual([s.id for s in defn.sections], ["sec_1"])
        self.assertEqual(
            [s.id for s in defn.all_section_groups()], ["sec_1", "sec_1.1"]
        )
        self.assertEqual(
            [g.id for g in defn.all_question_groups()], ["qg_1.1", "qg_1.1.1"]
        )
        self.assertEqual(
            [g.strategy for g in defn.all_question_groups()],
            ["all_required", "sequential"],
        )
        self.assertEqual(
            [q.id for q in defn.all_questions()],
            ["sq_1.1.1", "sq_1.1.2", "sq_1.1.1.1", "sq_1.1.1.2"],
        )
        self.assertEqual(
            [c.id for c in defn.all_criteria()],
            ["c_yes", "c_high", "c_low", "c_at_low", "c_mixed"],
        )
        # the conditional jump survives the round trip
        jump = defn.question_by_id("sq_1.1.2").jumps[0]
        self.assertEqual(jump.to, "sq_1.1.1")
        self.assertEqual(jump.on_criterion, "c_yes")
        self.assertIs(jump.when, CriterionStatus.SATISFIED)
        # the composite rule AST is preserved (all -> [any -> [...], not -> ...])
        mixed = defn.criterion_by_id("c_mixed").rule
        self.assertEqual(mixed.op, "all")
        self.assertEqual(mixed.rules[0].op, "any")
        self.assertEqual(mixed.rules[1].op, "not")
        # value declarations round trip
        score = next(v for v in defn.values if v.name == "score")
        self.assertEqual(score.thresholds, {"low": 3.0, "high": 8.0})

    def test_default_group_strategy_is_all_required(self):
        # the safe default: without an explicit strategy a group must ask
        # every required question (the legacy adapter opts into
        # sequential explicitly, so legacy guides keep their order)
        group = QuestionGroup(
            id="qg_x", title="T", questions=[QuestionDefinition(id="sq_x", text="x")]
        )
        self.assertEqual(group.strategy, "all_required")

    def test_default_answer_type_is_string(self):
        question = QuestionDefinition(id="sq_x", text="x")
        self.assertEqual(question.answer_type, "string")
        self.assertTrue(question.required)

    def test_answer_types_roundtrip(self):
        defn = self._load_v2(VALID_V2_YAML)
        self.assertEqual(defn.question_by_id("sq_1.1.1").answer_type, "string")
        self.assertEqual(defn.question_by_id("sq_1.1.2").answer_type, "boolean")
        self.assertEqual(defn.question_by_id("sq_1.1.1.2").answer_type, "number")
        self.assertFalse(defn.question_by_id("sq_1.1.1.2").required)

    def test_loops_are_allowed(self):
        # VALID_V2_YAML contains the loop sq_1.1.2 -> sq_1.1.1 (same
        # group); loops are allowed by design, so the file loads and the
        # jump is kept as data
        defn = self._load_v2(VALID_V2_YAML)
        jump = defn.question_by_id("sq_1.1.2").jumps[0]
        self.assertEqual(jump.to, "sq_1.1.1")

    def test_jump_targets_can_be_question_group_or_section(self):
        defn = self._load_v2(VALID_V2_YAML)
        self.assertEqual(defn.question_by_id("sq_1.1.2").jumps[1].to, "qg_1.1.1")
        self.assertEqual(defn.question_by_id("sq_1.1.1.1").jumps[0].to, "sec_1")
        self.assertIsNone(defn.question_by_id("sq_1.1.1.1").jumps[0].on_criterion)

    def test_section_relation_roundtrip(self):
        defn = self._load_v2(VALID_V2_YAML)
        nested = defn.section_by_id("sec_1.1")
        self.assertEqual(nested.relations[0].to, "sec_1")
        self.assertEqual(nested.relations[0].kind, "temporal")
        self.assertEqual(
            nested.relations[0].note, "nested section relates to its parent"
        )

    def test_v2_example_in_repo_loads(self):
        defn = load_interview_definition(
            os.path.join(INTERVIEWS_DIR, "depression_scid_v2_example.yaml")
        )
        self.assertEqual(defn.definition_version, "scid_depression_v2@2")
        self.assertEqual([s.id for s in defn.sections], ["sec_1", "sec_2"])
        self.assertEqual(
            [g.id for g in defn.all_question_groups()],
            ["qg_1.1", "qg_1.1.1", "qg_2.1"],
        )
        self.assertEqual(
            [q.id for q in defn.all_questions()],
            ["sq_1.1.1", "sq_1.1.2", "sq_1.1.1.1", "sq_1.1.1.2", "sq_2.1.1"],
        )
        # the nested section shows up in the walk at the right depth
        paths = [path for path, _ in defn._walk()]
        self.assertIn("sec_1 > sec_1.1", paths)
        self.assertTrue(defn.canonical_fallback_message)

    def test_auto_detect_v2_and_legacy(self):
        v2 = load_interview_definition(
            os.path.join(INTERVIEWS_DIR, "depression_scid_v2_example.yaml")
        )
        legacy = load_interview_definition(
            os.path.join(INTERVIEWS_DIR, "depression_scid_example.yaml")
        )
        # v2 files load natively, legacy files through the adapter
        self.assertNotIn("@legacy-", v2.definition_version)
        self.assertIn("@legacy-", legacy.definition_version)
        # the legacy adapter generates v2-convention question ids
        self.assertEqual(
            [q.id for q in legacy.all_questions()],
            ["sq_1.1.1", "sq_1.1.2", "sq_2.1.1"],
        )

    def test_strict_v2_rejects_guide_without_questions(self):
        import yaml

        data = yaml.safe_load(VALID_V2_YAML)
        data["sections"] = []
        self._expect_error(yaml.safe_dump(data), "definition has no questions")

    def test_group_without_questions_rejected(self):
        import yaml

        data = yaml.safe_load(VALID_V2_YAML)
        data["sections"][0]["children"][0]["questions"] = []
        self._expect_error(yaml.safe_dump(data), "at least 1 item")

    def test_unknown_top_level_field_is_rejected(self):
        self._expect_error(
            VALID_V2_YAML + "\nsome_new_field: 1", "some_new_field"
        )

    def test_definitions_are_frozen(self):
        defn = self._load_v2(VALID_V2_YAML)
        with self.assertRaises(ValidationError):
            defn.title = "changed"  # type: ignore[misc]


class TestIdPrefixes(DefinitionTestBase):
    """The pragmatic id rule: prefixes are mandatory, numeric parts are not."""

    def test_question_id_requires_sq_prefix(self):
        with self.assertRaises(ValidationError) as ctx:
            QuestionDefinition(id="q_1", text="x")
        self.assertIn(
            "question id 'q_1' must start with 'sq_'", str(ctx.exception)
        )

    def test_question_group_id_requires_qg_prefix(self):
        with self.assertRaises(ValidationError) as ctx:
            QuestionGroup(
                id="g_1",
                title="T",
                questions=[QuestionDefinition(id="sq_1", text="x")],
            )
        self.assertIn(
            "question group id 'g_1' must start with 'qg_'", str(ctx.exception)
        )

    def test_section_id_requires_sec_prefix(self):
        with self.assertRaises(ValidationError) as ctx:
            SectionGroup(id="sec", title="S")
        self.assertIn("section id 'sec' must start with 'sec_'", str(ctx.exception))

    def test_numeric_parts_are_informational_only(self):
        # positions are assigned at creation and never re-validated, so
        # any stable id after the prefix is accepted (no renumbering)
        group = QuestionGroup(
            id="qg_42.999",
            title="T",
            questions=[QuestionDefinition(id="sq_42.999.1", text="x")],
        )
        section = SectionGroup(id="sec_9.8.7", title="S", children=[group])
        defn = InterviewDefinition(
            definition_version="v",
            guide_id="g",
            title="t",
            sections=[section],
        )
        self.assertEqual(defn.section_by_id("sec_9.8.7"), section)
        self.assertEqual(defn.group_by_id("qg_42.999"), group)
        self.assertEqual(defn.question_by_id("sq_42.999.1").text, "x")


class TestCriteriaGroups(DefinitionTestBase):
    """Criteria live on criteria groups (question OR section groups)."""

    def test_criteria_without_marker_rejected(self):
        group = QuestionGroup(
            id="qg_1",
            title="T",
            questions=[QuestionDefinition(id="sq_1", text="x")],
            criteria=[CriterionDefinition(id="c1", rule=EqRule(value="v", equals=True))],
        )
        # needs a declared value, otherwise the rule check adds a second error
        with self.assertRaises(ValidationError) as ctx:
            InterviewDefinition(
                definition_version="v",
                guide_id="g",
                title="t",
                values=[ValueDeclaration(name="v", type="bool")],
                sections=[SectionGroup(id="sec_1", title="S", children=[group])],
            )
        self.assertIn(
            "has criteria but is not marked as a criteria group", str(ctx.exception)
        )

    def test_marker_without_criteria_rejected(self):
        group = QuestionGroup(
            id="qg_1",
            title="T",
            criteria_group=True,
            questions=[QuestionDefinition(id="sq_1", text="x")],
        )
        with self.assertRaises(ValidationError) as ctx:
            InterviewDefinition(
                definition_version="v",
                guide_id="g",
                title="t",
                sections=[SectionGroup(id="sec_1", title="S", children=[group])],
            )
        self.assertIn("marked as a criteria group but has no criteria", str(ctx.exception))

    def test_section_group_can_be_a_criteria_group(self):
        section = SectionGroup(
            id="sec_1",
            title="S",
            criteria_group=True,
            criteria=[CriterionDefinition(id="c1", rule=EqRule(value="v", equals=True))],
            children=[
                QuestionGroup(
                    id="qg_1",
                    title="T",
                    questions=[QuestionDefinition(id="sq_1", text="x")],
                )
            ],
        )
        defn = InterviewDefinition(
            definition_version="v",
            guide_id="g",
            title="t",
            values=[ValueDeclaration(name="v", type="bool")],
            sections=[section],
        )
        self.assertEqual([c.id for c in defn.all_criteria()], ["c1"])
        self.assertEqual(defn.criterion_by_id("c1").id, "c1")

    def test_criteria_collected_in_document_order(self):
        defn = self._load_v2(VALID_V2_YAML)
        self.assertEqual(
            [c.id for c in defn.all_criteria()],
            ["c_yes", "c_high", "c_low", "c_at_low", "c_mixed"],
        )


class TestLegacyAdapter(DefinitionTestBase):
    """to_v2() maps the legacy format onto the v2 model without loss."""

    def _legacy_data(self):
        import yaml

        return yaml.safe_load(LEGACY_YAML)

    def test_preserves_order_flags_and_profile(self):
        defn = to_v2(self._legacy_data(), "legacy_test.yaml")
        self.assertEqual(defn.guide_id, "legacy_test")
        self.assertEqual(defn.behavior_profile, "strict_clinical")
        self.assertEqual(defn.language, "de")
        self.assertEqual(defn.framing_message, "Welcome.")  # trimmed
        self.assertEqual(
            [(s.id, s.title) for s in defn.sections],
            [("sec_1", "Section A"), ("sec_2", "Section B")],
        )
        # each legacy section becomes one sequential question group
        groups = defn.all_question_groups()
        self.assertEqual([g.id for g in groups], ["qg_1.1", "qg_2.1"])
        self.assertEqual({g.strategy for g in groups}, {"sequential"})
        questions = {q.id: q for q in defn.all_questions()}
        self.assertEqual(
            list(questions), ["sq_1.1.1", "sq_1.1.2", "sq_2.1.1"]
        )  # legacy order
        self.assertTrue(questions["sq_1.1.1"].required)
        self.assertTrue(questions["sq_1.1.2"].allow_followup)
        self.assertFalse(questions["sq_2.1.1"].allow_followup)

    def test_legacy_question_ids_are_remapped(self):
        # the adapter generates v2-convention ids; the legacy ids
        # (q1/q2/q3) no longer exist as question ids
        defn = to_v2(self._legacy_data(), "legacy_test.yaml")
        self.assertEqual(
            {q.id for q in defn.all_questions()},
            {"sq_1.1.1", "sq_1.1.2", "sq_2.1.1"},
        )
        for legacy_id in ("q1", "q2", "q3"):
            self.assertIsNone(defn.question_by_id(legacy_id))

    def test_definition_version_stable_and_content_sensitive(self):
        data = self._legacy_data()
        a = to_v2(data, "f.yaml")
        b = to_v2(copy.deepcopy(data), "f.yaml")
        self.assertEqual(a.definition_version, b.definition_version)
        self.assertRegex(a.definition_version, r"^legacy_test@legacy-[0-9a-f]{8}$")
        changed = copy.deepcopy(data)
        changed["sections"][0]["questions"][0]["text"] = "changed!"
        c = to_v2(changed, "f.yaml")
        self.assertNotEqual(a.definition_version, c.definition_version)

    def test_missing_guide_id_falls_back_to_filename(self):
        data = self._legacy_data()
        del data["guide_id"]
        defn = to_v2(data, "my_file")
        self.assertEqual(defn.guide_id, "my_file")

    def test_missing_guide_id_and_filename_falls_back(self):
        # matches the old loader: guide_id falls back to the file name
        # given to the loader; with no name at all, a neutral id is used
        data = self._legacy_data()
        del data["guide_id"]
        defn = to_v2(data, "")
        self.assertEqual(defn.guide_id, "legacy_guide")

    def test_missing_question_id_or_text_fails_loudly(self):
        data = self._legacy_data()
        del data["sections"][0]["questions"][0]["id"]
        with self.assertRaises(InterviewDefinitionError) as ctx:
            to_v2(data, "f.yaml")
        self.assertIn("has no 'id'", str(ctx.exception))
        data = self._legacy_data()
        del data["sections"][1]["questions"][0]["text"]
        with self.assertRaises(InterviewDefinitionError):
            to_v2(data, "f.yaml")

    def test_empty_legacy_guide_still_loads(self):
        data = self._legacy_data()
        data["sections"] = []
        defn = to_v2(data, "f.yaml")
        self.assertEqual(defn.sections, [])
        self.assertTrue(defn.definition_version.startswith("legacy_test@legacy-"))

    def test_load_legacy_guide_reads_file(self):
        import interview_definition

        path = self._write(LEGACY_YAML)
        defn = interview_definition.load_legacy_guide(path)
        self.assertEqual(
            [q.id for q in defn.all_questions()], ["sq_1.1.1", "sq_1.1.2", "sq_2.1.1"]
        )


class TestInterviewGuideWrapper(DefinitionTestBase):
    """The legacy InterviewGuide wrapper stays compatible with app.py."""

    def test_wrapper_exposes_v2_definition(self):
        g = InterviewGuide("depression_scid_example.yaml")
        self.assertIsInstance(g.definition, InterviewDefinition)
        self.assertTrue(g.definition_version.startswith("depression_scid_example@legacy-"))
        # the wrapper flattens the hierarchy to the legacy question dicts
        # (generated adapter ids, legacy order)
        self.assertEqual(
            [q["id"] for q in g.questions], ["sq_1.1.1", "sq_1.1.2", "sq_2.1.1"]
        )
        self.assertEqual(
            {q["section_title"] for q in g.questions}, {"Einstieg", "Schlaf"}
        )

    def test_v2_file_rejected_by_legacy_wrapper(self):
        with self.assertRaises(InterviewDefinitionError) as ctx:
            InterviewGuide("depression_scid_v2_example.yaml")
        self.assertIn("v2", str(ctx.exception))

    def test_unknown_profile_fails_fast(self):
        path = self._write(
            LEGACY_YAML.replace(
                "behavior_profile: strict_clinical", "behavior_profile: nope"
            )
        )
        with self.assertRaises(InterviewDefinitionError) as ctx:
            InterviewGuide(os.path.basename(path))
        self.assertIn("profiles.yaml", str(ctx.exception))

    def test_guide_file_still_in_interviews_dir(self):
        # legacy files are discovered by filename, as before
        g = InterviewGuide("depression_scid_example.yaml")
        self.assertEqual(
            g.title, "Example: SCID Depression Module (shortened, for testing)"
        )


class TestImportValidation(DefinitionTestBase):
    """All cross-reference checks fail loudly with aggregated messages."""

    def test_duplicate_question_id(self):
        self._expect_error(
            VALID_V2_YAML.replace("id: sq_1.1.1.1", "id: sq_1.1.1"),
            "duplicate question id 'sq_1.1.1'",
        )

    def test_duplicate_group_id(self):
        self._expect_error(
            VALID_V2_YAML.replace("id: qg_1.1.1", "id: qg_1.1"),
            "duplicate group id 'qg_1.1'",
        )

    def test_duplicate_section_id(self):
        self._expect_error(
            VALID_V2_YAML.replace("id: sec_1.1", "id: sec_1"),
            "duplicate section id 'sec_1'",
        )

    def test_duplicate_value_name(self):
        self._expect_error(
            VALID_V2_YAML.replace("name: pattern", "name: yes_no"),
            "duplicate value id 'yes_no'",
        )

    def test_duplicate_criterion_id(self):
        self._expect_error(
            VALID_V2_YAML.replace("id: c_high", "id: c_yes"),
            "duplicate criterion id 'c_yes'",
        )

    def test_duplicate_meaning_unit_id(self):
        text = VALID_V2_YAML.replace(
            '              - id: sq_1.1.1.1\n                text: "Q three?"\n',
            '              - id: sq_1.1.1.1\n                text: "Q three?"\n'
            '                meaning_units:\n'
            '                  - id: mu_one\n'
            '                    description: "clash"\n',
        )
        self._expect_error(text, "duplicate meaning-unit id 'mu_one'")

    def test_jump_to_unknown_target(self):
        self._expect_error(
            VALID_V2_YAML.replace("to: qg_1.1.1", "to: qg_zzz"),
            "jump target 'qg_zzz' is not a known section, group or question id",
        )

    def test_jump_on_unknown_criterion(self):
        self._expect_error(
            VALID_V2_YAML.replace("on_criterion: c_yes", "on_criterion: c_zzz"),
            "on_criterion 'c_zzz' is not a declared criterion",
        )

    def test_relation_to_unknown_section(self):
        self._expect_error(
            VALID_V2_YAML.replace(
                "to: sec_1\n            kind: temporal",
                "to: sec_zzz\n            kind: temporal",
            ),
            "relation target 'sec_zzz' is not a known section id",
        )

    def test_relation_to_self_rejected(self):
        self._expect_error(
            VALID_V2_YAML.replace(
                "to: sec_1\n            kind: temporal",
                "to: sec_1.1\n            kind: temporal",
            ),
            "a section cannot relate to itself",
        )

    def test_criteria_without_criteria_group_marker(self):
        text = VALID_V2_YAML.replace("criteria_group: true", "", 1)
        self._expect_error(
            text, "has criteria but is not marked as a criteria group"
        )

    def test_unknown_v2_profile_rejected(self):
        self._expect_error(
            VALID_V2_YAML.replace(
                "behavior_profile: strict_clinical", "behavior_profile: nope"
            ),
            "not defined in study_content/prompts/profiles.yaml",
        )

    def test_legacy_profile_check_applies_after_adaptation(self):
        import interview_definition

        path = self._write(
            LEGACY_YAML.replace(
                "behavior_profile: strict_clinical", "behavior_profile: nope"
            )
        )
        with self.assertRaises(InterviewDefinitionError) as ctx:
            interview_definition.load_interview_definition(path)
        self.assertIn(
            "not defined in study_content/prompts/profiles.yaml",
            str(ctx.exception),
        )

    def test_criterion_references_unknown_value(self):
        self._expect_error(
            VALID_V2_YAML.replace(
                "rule: {op: eq, value: yes_no, equals: true}",
                "rule: {op: eq, value: nope, equals: true}",
            ),
            "value 'nope' is not declared in 'values'",
        )

    def test_unsupported_language_rejected(self):
        self._expect_error(
            VALID_V2_YAML.replace("language: de", "language: xx"),
            "language 'xx' is not supported",
        )

    # -- rule / value type checks via direct model construction --

    def test_gte_on_bool_value(self):
        self._expect_rule_error(
            [{"name": "yes_no", "type": "bool"}],
            [GteRule(value="yes_no", threshold=1)],
            "gte needs a number value, but 'yes_no' is declared as bool",
        )

    def test_eq_type_mismatch(self):
        self._expect_rule_error(
            [{"name": "yes_no", "type": "bool"}],
            [EqRule(value="yes_no", equals="yes")],
            "does not match the declared type of value 'yes_no' (type: bool)",
        )

    def test_eq_category_outside_declared_categories(self):
        self._expect_rule_error(
            [{"name": "pattern", "type": "category", "categories": ["a", "b"]}],
            [EqRule(value="pattern", equals="z")],
            "does not match the declared type of value 'pattern' "
            "(type: category, categories: ['a', 'b'])",
        )

    def test_composite_rules_checked_recursively(self):
        # the nested 'any' -> 'eq' chain is type-checked too
        self._expect_rule_error(
            [{"name": "yes_no", "type": "bool"}],
            [AnyRule(rules=[NotRule(rule=EqRule(value="yes_no", equals="yes"))])],
            "does not match the declared type",
        )

    def test_category_value_without_categories(self):
        self._expect_rule_error(
            [{"name": "p", "type": "category"}],
            [],
            "need at least one 'categories' entry",
        )

    def test_number_value_min_gt_max(self):
        self._expect_rule_error(
            [{"name": "n", "type": "number", "min": 5, "max": 1}],
            [],
            "must not exceed max",
        )

    def test_threshold_not_declared_on_value(self):
        self._expect_rule_error(
            [{"name": "score", "type": "number", "thresholds": {"low": 3}}],
            [ThresholdRule(value="score", threshold="mid")],
            "threshold 'mid' is not declared on value 'score' "
            "(declared thresholds: ['low'])",
        )

    def test_multiple_problems_reported_in_one_pass(self):
        # two model-level problems: duplicate question id + unknown jump target
        text = VALID_V2_YAML.replace(
            "id: sq_1.1.1.1", "id: sq_1.1.1"
        ).replace("to: qg_1.1.1", "to: qg_zzz")
        with self.assertRaises(InterviewDefinitionError) as ctx:
            self._load_v2(text)
        message = str(ctx.exception)
        self.assertIn("sq_1.1.1", message)
        self.assertIn("qg_zzz", message)
