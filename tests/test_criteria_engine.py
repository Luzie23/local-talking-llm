"""
Unit tests for backend/criteria_engine.py (RECONSTRUCTION_PLAN.md
Phase 2, step 2.3): the rule engine for criteria groups — the full
operator set (logical not/all/any, arithmetic eq/neq/gte/lte/between/
min/max/threshold, text contains/not_contains, completeness
is_empty/is_not_empty, group-level count/majority/weighted_sum) over
the FOUR-VALUED criterion logic (SATISFIED / NOT_SATISFIED / OPEN /
UNCLEAR), including the reading verdicts (missing -> OPEN, unreliable
-> UNCLEAR, type mismatch -> UNCLEAR) and the logical truth tables.

All tests are deterministic and need no network access.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from contracts import CriterionStatus, ValueReading
from criteria_engine import (
    CriteriaEvaluationError,
    evaluate_criteria,
    evaluate_rule,
)
from interview_definition import (
    AllRule,
    AnyRule,
    BetweenRule,
    CountRule,
    CriterionDefinition,
    ContainsRule,
    EqRule,
    GteRule,
    InterviewDefinition,
    IsEmptyRule,
    IsNotEmptyRule,
    LteRule,
    MajorityRule,
    MaxRule,
    MinRule,
    NeqRule,
    NotContainsRule,
    NotRule,
    QuestionDefinition,
    QuestionGroup,
    SectionGroup,
    ThresholdRule,
    ValueDeclaration,
    WeightedSumRule,
)

S = CriterionStatus.SATISFIED
NS = CriterionStatus.NOT_SATISFIED
O = CriterionStatus.OPEN
U = CriterionStatus.UNCLEAR

# four declared values, one per type
DECLS = {
    "flag": ValueDeclaration(name="flag", type="bool"),
    "score": ValueDeclaration(
        name="score", type="number", min=0, max=10,
        thresholds={"low": 3, "high": 8},
    ),
    "cat": ValueDeclaration(name="cat", type="category", categories=["a", "b"]),
    "text": ValueDeclaration(name="text", type="string"),
    # group-level aggregation values (criteria-group elements)
    "b1": ValueDeclaration(name="b1", type="bool"),
    "b2": ValueDeclaration(name="b2", type="bool"),
    "b3": ValueDeclaration(name="b3", type="bool"),
    "n1": ValueDeclaration(name="n1", type="number"),
    "n2": ValueDeclaration(name="n2", type="number"),
}


def R(name: str, value, reliable: bool = True) -> ValueReading:
    return ValueReading(name=name, value=value, reliable=reliable)


def readings(**kwargs) -> dict[str, ValueReading]:
    """Build a readings mapping from keyword name=value / name=reliably(value)."""
    out = {}
    for key, val in kwargs.items():
        if isinstance(val, ValueReading):
            out[key] = val
        else:
            out[key] = R(key, val)
    return out


class TestLeafReadingVerdicts(unittest.TestCase):
    """The four-valued reading verdicts: missing -> OPEN, unreliable and
    type-mismatched -> UNCLEAR, valid -> SATISFIED/NOT_SATISFIED."""

    def test_missing_value_is_open(self):
        for rule in (
            EqRule(value="flag", equals=True),
            GteRule(value="score", threshold=5),
            ContainsRule(value="text", needle="x"),
            IsEmptyRule(value="text"),
            CountRule(values=["flag"], if_eq=True, gte=1),
        ):
            with self.subTest(rule=rule.op):
                self.assertEqual(evaluate_rule(rule, {}, DECLS), O)

    def test_unreliable_reading_is_unclear(self):
        for name, rule in (
            ("flag", EqRule(value="flag", equals=True)),
            ("score", GteRule(value="score", threshold=5)),
            ("text", ContainsRule(value="text", needle="x")),
            ("text", IsNotEmptyRule(value="text")),
        ):
            with self.subTest(rule=rule.op):
                self.assertEqual(
                    evaluate_rule(rule, {name: R(name, None, reliable=False)}, DECLS),
                    U,
                )

    def test_type_mismatch_is_unclear_not_not_satisfied(self):
        self.assertEqual(
            evaluate_rule(EqRule(value="flag", equals=True),
                          readings(flag="ja"), DECLS),
            U,
        )
        self.assertEqual(
            evaluate_rule(GteRule(value="score", threshold=5),
                          readings(score="7"), DECLS),
            U,
        )
        # a bool is not a number
        self.assertEqual(
            evaluate_rule(GteRule(value="score", threshold=5),
                          readings(score=True), DECLS),
            U,
        )

    def test_valid_readings_decide(self):
        self.assertEqual(
            evaluate_rule(EqRule(value="flag", equals=True),
                          readings(flag=True), DECLS),
            S,
        )
        self.assertEqual(
            evaluate_rule(EqRule(value="flag", equals=True),
                          readings(flag=False), DECLS),
            NS,
        )
        self.assertEqual(
            evaluate_rule(ContainsRule(value="text", needle="x"),
                          readings(text=""), DECLS),
            NS,
        )


class TestArithmeticOperators(unittest.TestCase):
    """eq/neq/gte/lte/between/min/max/threshold — boundaries included
    (gte/lte/between use >= and <=)."""

    def test_eq_neq(self):
        self.assertEqual(
            evaluate_rule(NeqRule(value="flag", not_equals=False),
                          readings(flag=True), DECLS),
            S,
        )
        self.assertEqual(
            evaluate_rule(NeqRule(value="score", not_equals=5),
                          readings(score=6), DECLS),
            S,
        )
        self.assertEqual(
            evaluate_rule(NeqRule(value="cat", not_equals="a"),
                          readings(cat="a"), DECLS),
            NS,
        )

    def test_comparisons_inclusive(self):
        self.assertEqual(evaluate_rule(GteRule(value="score", threshold=5),
                                       readings(score=5), DECLS), S)
        self.assertEqual(evaluate_rule(GteRule(value="score", threshold=5),
                                       readings(score=4.99), DECLS), NS)
        self.assertEqual(evaluate_rule(LteRule(value="score", threshold=5),
                                       readings(score=5), DECLS), S)
        self.assertEqual(evaluate_rule(LteRule(value="score", threshold=5),
                                       readings(score=5.01), DECLS), NS)

    def test_between_inclusive(self):
        rule = BetweenRule(value="score", min=3, max=8)
        self.assertEqual(evaluate_rule(rule, readings(score=3), DECLS), S)
        self.assertEqual(evaluate_rule(rule, readings(score=8), DECLS), S)
        self.assertEqual(evaluate_rule(rule, readings(score=5), DECLS), S)
        self.assertEqual(evaluate_rule(rule, readings(score=2.9), DECLS), NS)
        self.assertEqual(evaluate_rule(rule, readings(score=8.1), DECLS), NS)

    def test_min_max(self):
        self.assertEqual(evaluate_rule(MinRule(value="score", threshold=3),
                                       readings(score=3), DECLS), S)
        self.assertEqual(evaluate_rule(MinRule(value="score", threshold=3),
                                       readings(score=2.999), DECLS), NS)
        self.assertEqual(evaluate_rule(MaxRule(value="score", threshold=8),
                                       readings(score=8), DECLS), S)
        self.assertEqual(evaluate_rule(MaxRule(value="score", threshold=8),
                                       readings(score=8.001), DECLS), NS)

    def test_threshold_named_and_exact(self):
        # threshold semantics: value >= the NAMED threshold (declared on
        # the value, not a magic number in the rule)
        self.assertEqual(
            evaluate_rule(ThresholdRule(value="score", threshold="high"),
                          readings(score=8), DECLS),
            S,
        )
        self.assertEqual(
            evaluate_rule(ThresholdRule(value="score", threshold="high"),
                          readings(score=7.9), DECLS),
            NS,
        )
        self.assertEqual(
            evaluate_rule(ThresholdRule(value="score", threshold="low"),
                          readings(score=3), DECLS),
            S,
        )
        self.assertEqual(
            evaluate_rule(ThresholdRule(value="score", threshold="low"),
                          readings(score=2.9), DECLS),
            NS,
        )

    def test_threshold_unknown_name_raises(self):
        with self.assertRaises(CriteriaEvaluationError):
            evaluate_rule(ThresholdRule(value="score", threshold="nope"),
                          readings(score=9), DECLS)


def all_of(*statuses):
    """The documented all-semantics (reference for the tests)."""
    if NS in statuses:
        return NS
    if U in statuses:
        return U
    if O in statuses:
        return O
    return S


def any_of(*statuses):
    """The documented any-semantics (reference for the tests)."""
    if S in statuses:
        return S
    if U in statuses:
        return U
    if O in statuses:
        return O
    return NS


class TestLogicalTruthTables(unittest.TestCase):
    """not/all/any over the four-valued logic, exact tables.

    Each child's status is driven independently by its own reading:
      flag=True -> S, flag=False -> NS, flag missing -> O, unreliable -> U
      (same for the category value 'cat').
    """

    A = EqRule(value="flag", equals=True)
    B = EqRule(value="cat", equals="a")

    def m(self, case_a: str, case_b: str) -> dict[str, ValueReading]:
        """Readings such that child A gets case_a and child B gets case_b."""
        out: dict[str, ValueReading] = {}
        for key, case, sat_value in (
            ("flag", case_a, True),
            ("cat", case_b, "a"),
        ):
            if case == "S":
                out[key] = R(key, sat_value)
            elif case == "NS":
                out[key] = R(key, "b" if key == "cat" else False)
            elif case == "U":
                out[key] = R(key, None, reliable=False)
            # "O": absent
        return out

    def verify_table(self, rule, combine):
        for ca, want_a in (("S", S), ("NS", NS), ("O", O), ("U", U)):
            for cb, want_b in (("S", S), ("NS", NS), ("O", O), ("U", U)):
                m = self.m(ca, cb)
                self.assertEqual(evaluate_rule(rule.rules[0], m, DECLS),
                                 want_a, f"{rule.op} child A {ca}")
                self.assertEqual(evaluate_rule(rule.rules[1], m, DECLS),
                                 want_b, f"{rule.op} child B {cb}")
                got = evaluate_rule(rule, m, DECLS)
                self.assertEqual(got, combine(want_a, want_b),
                                 f"{rule.op}({ca},{cb}) got {got}")

    def test_not_table(self):
        expected = {"S": NS, "NS": S, "O": O, "U": U}
        for case, want in expected.items():
            got = evaluate_rule(NotRule(rule=self.A), self.m(case, "O"), DECLS)
            self.assertEqual(got, want, f"not({case})")

    def test_all_table(self):
        self.verify_table(AllRule(rules=[self.A, self.B]), all_of)

    def test_any_table(self):
        self.verify_table(AnyRule(rules=[self.A, self.B]), any_of)

    def test_three_way_mixes(self):
        a, b = self.A, self.B
        c = EqRule(value="text", equals="x")
        all3 = AllRule(rules=[a, b, c])
        self.assertEqual(evaluate_rule(all3, {}, DECLS), O)
        self.assertEqual(
            evaluate_rule(all3, {"flag": R("flag", None, reliable=False)},
                          DECLS),
            U,  # UNCLEAR beats OPEN
        )
        self.assertEqual(
            evaluate_rule(AnyRule(rules=[a, b, c]),
                          {"flag": R("flag", True)}, DECLS),
            S)
        self.assertEqual(
            evaluate_rule(all3, {"flag": R("flag", False)}, DECLS), NS)
        # not(all3) with one child NOT_SAT: all3 -> NS, not(NS) -> S
        self.assertEqual(
            evaluate_rule(NotRule(rule=all3),
                          {"flag": R("flag", False)}, DECLS), S)


class TestTextAndCompleteness(unittest.TestCase):
    """contains/not_contains are case-insensitive; is_empty also treats
    None / blank strings as empty."""

    def test_contains_case_insensitive(self):
        rule = ContainsRule(value="text", needle="Schlaf")
        self.assertEqual(evaluate_rule(rule, readings(text="kein SCHLAF"),
                                       DECLS), S)
        self.assertEqual(evaluate_rule(rule, readings(text="gut"),
                                       DECLS), NS)

    def test_not_contains(self):
        rule = NotContainsRule(value="text", needle="gut")
        self.assertEqual(evaluate_rule(rule, readings(text="gut"),
                                       DECLS), NS)
        self.assertEqual(evaluate_rule(rule, readings(text="GUT, aber"),
                                       DECLS), NS)  # CI
        self.assertEqual(evaluate_rule(rule, readings(text="schlecht"),
                                       DECLS), S)

    def test_is_empty(self):
        rule = IsEmptyRule(value="text")
        for v in (None, "", "   "):
            with self.subTest(v=v):
                self.assertEqual(evaluate_rule(rule, readings(text=v), DECLS), S)
        self.assertEqual(evaluate_rule(rule, readings(text="x"), DECLS), NS)

    def test_is_not_empty(self):
        rule = IsNotEmptyRule(value="text")
        for v in (None, "", "   "):
            with self.subTest(v=v):
                self.assertEqual(evaluate_rule(rule, readings(text=v), DECLS),
                                 NS)
        self.assertEqual(evaluate_rule(rule, readings(text="x"), DECLS), S)

    def test_is_empty_on_number_present(self):
        # the reading exists and its type matches -> not empty
        self.assertEqual(
            evaluate_rule(IsEmptyRule(value="score"),
                          readings(score=0), DECLS),
            NS,
        )


class TestAggregation(unittest.TestCase):
    """count / majority / weighted_sum — group level: the listed values
    (the criteria group's elements) are evaluated TOGETHER; one missing
    value -> OPEN, one unreliable or type-mismatched value -> UNCLEAR;
    only a fully evaluable group is counted/summed."""

    def drive(self, **kwargs) -> dict[str, ValueReading]:
        return {k: (v if isinstance(v, ValueReading) else R(k, v))
                for k, v in kwargs.items()}

    def test_count(self):
        rule = CountRule(values=["b1", "b2", "b3"], if_eq=True, gte=2)
        self.assertEqual(
            evaluate_rule(rule, self.drive(b1=True, b2=True, b3=False),
                          DECLS), S)  # 2 >= 2
        self.assertEqual(
            evaluate_rule(rule, self.drive(b1=True, b2=False, b3=False),
                          DECLS), NS)  # 1 < 2
        self.assertEqual(evaluate_rule(rule, self.drive(b1=True), DECLS),
                         O)  # b2/b3 missing
        self.assertEqual(
            evaluate_rule(rule, self.drive(
                b1=R("b1", None, reliable=False), b2=True, b3=True),
                          DECLS), U)

    def test_count_type_mismatch_is_unclear(self):
        rule = CountRule(values=["b1"], if_eq=True, gte=1)
        self.assertEqual(evaluate_rule(rule, self.drive(b1="ja"),
                                       DECLS), U)

    def test_majority(self):
        rule = MajorityRule(values=["b1", "b2", "b3"], if_eq=True)
        self.assertEqual(
            evaluate_rule(rule, self.drive(b1=True, b2=True, b3=False),
                          DECLS), S)  # 2 > 1.5
        self.assertEqual(
            evaluate_rule(rule, self.drive(b1=True, b2=False, b3=False),
                          DECLS), NS)  # 1 < 1.5
        self.assertEqual(
            evaluate_rule(rule, self.drive(b1=True, b2=True, b3=True),
                          DECLS), S)
        self.assertEqual(evaluate_rule(rule, self.drive(b1=True), DECLS),
                         O)

    def test_majority_exact_half_is_not_satisfied(self):
        rule = MajorityRule(
            values=["b1", "b2", "b3", "flag"], if_eq=True)
        self.assertEqual(
            evaluate_rule(rule, self.drive(
                b1=True, b2=True, b3=False, flag=False), DECLS), NS)  # 2/4

    def test_weighted_sum(self):
        rule = WeightedSumRule(
            values=["n1", "n2"],
            weights={"n1": 2.0, "n2": 1.0},
            threshold=4.0,
        )
        self.assertEqual(evaluate_rule(rule, self.drive(n1=1, n2=2),
                                       DECLS), S)  # exactly 4.0
        self.assertEqual(evaluate_rule(rule, self.drive(n1=1, n2=1.9),
                                       DECLS), NS)  # 3.9 < 4
        self.assertEqual(evaluate_rule(rule, self.drive(n1=1), DECLS),
                         O)
        self.assertEqual(
            evaluate_rule(rule, self.drive(n1=R("n1", None,
                                                reliable=False), n2=2),
                          DECLS), U)

    def test_weighted_sum_missing_weight_raises(self):
        rule = WeightedSumRule(values=["n1", "n2"],
                               weights={"n1": 1.0}, threshold=1.0)
        with self.assertRaises(CriteriaEvaluationError):
            evaluate_rule(rule, self.drive(n1=1, n2=1), DECLS)


def _definition_with_criteria(criteria):
    """A minimal valid v2 definition carrying the given criteria."""
    section = SectionGroup(
        id="sec_1", title="S", criteria_group=True, criteria=criteria,
        children=[QuestionGroup(
            id="qg_1", title="Q", questions=[
                QuestionDefinition(id="sq_1.1", text="q?",
                                   answer_type="boolean"),
            ],
        )],
    )
    return InterviewDefinition(
        definition_version="2.0", guide_id="interview_x", title="X",
        language="de", sections=[section], values=list(DECLS.values()),
    )


class TestDefinitionLevel(unittest.TestCase):
    """evaluate_criteria over a full InterviewDefinition."""

    def test_all_open_with_no_readings(self):
        defs = [
            CriterionDefinition(
                id="c_a", description="d",
                rule=EqRule(value="flag", equals=True),
            ),
            CriterionDefinition(
                id="c_b", description="d",
                rule=AllRule(rules=[EqRule(value="flag", equals=True)]),
            ),
        ]
        got = evaluate_criteria(_definition_with_criteria(defs), {})
        self.assertEqual(got, {"c_a": O, "c_b": O})

    def test_decided_when_reading_present(self):
        defs = [CriterionDefinition(
            id="c_a", description="d",
            rule=EqRule(value="flag", equals=True),
        )]
        got = evaluate_criteria(
            _definition_with_criteria(defs),
            {"flag": R("flag", True)},
        )
        self.assertEqual(got, {"c_a": S})

    def test_unknown_value_rejected_at_import_time(self):
        # import-time validation rejects criteria referencing undeclared
        # values (so the engine never sees them through a definition)
        defs = [CriterionDefinition(
            id="c_a", description="d",
            rule=EqRule(value="nope", equals=True),
        )]
        with self.assertRaises(Exception):
            _definition_with_criteria(defs)

    def test_unknown_value_in_engine_raises(self):
        # the engine's own guard, for directly built rules (the value
        # must have a reading — a missing one is OPEN before the check):
        with self.assertRaises(CriteriaEvaluationError):
            evaluate_rule(EqRule(value="nope", equals=True),
                          {"nope": R("nope", True)},
                          {"flag": DECLS["flag"]})


if __name__ == "__main__":
    unittest.main()