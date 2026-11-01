"""
CRITERIA ENGINE — THE DECLARATIVE RULE EVALUATOR (PHASE 2, STEP 2.3)
=====================================================================
The rule engine for criteria groups: evaluates the declarative rule AST
(backend/interview_definition.py — the CriteriaRule union) against the
current value readings and produces the FOUR-VALUED criterion status
(SATISFIED / NOT_SATISFIED / OPEN / UNCLEAR — contracts.CriterionStatus).

PURE and deterministic: no LLM, no I/O, no state. It is a function of
(rule AST, value readings, value declarations). Everything else — who
extracts the values from the answers (the Diagnostiker, Phase 3), when
a criterion counts as "decided" and stops being re-evaluated (the
decided guard, applied by the Navigator, step 2.4), and which question
comes next (the Navigator) — is deliberately NOT this module's job.

Input — value readings (contracts.ValueReading), a mapping
value-name -> reading:
  * name NOT in the mapping — the question that provides the value has
    not been answered yet -> the rule contributes OPEN
  * reliable == False — an answer exists but could not be interpreted
    reliably -> the rule contributes UNCLEAR
  * reliable == True — the value is evaluated. `value` of None means
    the answer carried no extractable content ("empty"). A value that
    does not match the value's DECLARED type (bool / number / category
    / string) is not interpretable -> UNCLEAR — the engine never
    reports NOT_SATISFIED for an uninterpretable answer.

The four-valued logic (Kleene-like, with explicit priority):
    NOT :  SATISFIED <-> NOT_SATISFIED ; OPEN -> OPEN
           UNCLEAR -> UNCLEAR
    ALL :  NOT_SATISFIED if ANY child is NOT_SATISFIED
           SATISFIED     if ALL children are SATISFIED
           else          UNCLEAR if any child is UNCLEAR, else OPEN
    ANY :  SATISFIED     if ANY child is SATISFIED
           NOT_SATISFIED if ALL children are NOT_SATISFIED
           else          UNCLEAR if any child is UNCLEAR, else OPEN

A decided child dominates: in ALL one NOT_SATISFIED child decides the
whole rule (even if other children are still OPEN/UNCLEAR); in ANY one
SATISFIED child decides it. This is what makes the ONLINE character of
the interview work — a criterion becomes decided as soon as its
decisive answers are in (design note: criteria are evaluated
continuously during the conversation, as soon as the answers that feed
them are complete).

The operator set (the full set of docs/interview_navi_design.md):
  logical        not / all / any
  arithmetic     eq / neq / gte / lte / between / min / max / threshold
  text           contains / not_contains     (case-insensitive)
  completeness   is_empty / is_not_empty
  aggregation    count / majority / weighted_sum   (group level: the
                 listed values are the values set from the criteria
                 group's questions together)

"Decided" (a criterion SATISFIED/NOT_SATISFIED is not re-evaluated;
reopening only through UNCLEAR) is NOT enforced here — the engine is
purely functional. The Navigator applies that guard when it proposes
criterion updates to the state machine (interview_navigator.py).

Used by: backend/interview_navigator.py (step 2.4), later the
Supervisor (step 2.5), and the tests (step 2.6). The legacy pipeline
does not use this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contracts import CriterionStatus, ValueReading
from interview_definition import (
    AllRule,
    AnyRule,
    BetweenRule,
    CountRule,
    CriterionDefinition,
    CriteriaRule,
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
    ThresholdRule,
    ValueDeclaration,
    WeightedSumRule,
)


class CriteriaEvaluationError(ValueError):
    """
    A rule was evaluated that cannot be evaluated (a value name that
    import validation would have rejected, a named threshold that does
    not exist on the value, a missing weight in a weighted_sum). These
    are definition-construction errors, not interview states — they
    fail loudly instead of producing a criterion status.
    """


def _declared_type_ok(decl: ValueDeclaration, value: Any) -> bool:
    """
    Does a raw reading value match the value's declared type?
    None (the answer carried no extractable content) is a VALID reading
    for every type — it is "empty", not a type error; the operators
    then decide what empty means (is_empty -> SATISFIED, arithmetic ->
    UNCLEAR, see _leaf_single_value).
    """
    if value is None:
        return True
    if decl.type == "bool":
        return isinstance(value, bool)
    if decl.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    # "string" and "category" values carry raw text
    return isinstance(value, str)


def _reading(
    name: str, readings: Mapping[str, ValueReading]
) -> tuple[ValueReading | None, CriterionStatus | None]:
    """
    One value's reading and its immediate four-valued verdict:
    (None, OPEN) if the value has no reading yet, (reading, UNCLEAR) if
    the reading exists but is unreliable, (reading, None) if the value
    is evaluable.
    """
    reading = readings.get(name)
    if reading is None:
        return None, CriterionStatus.OPEN
    if not reading.reliable:
        return reading, CriterionStatus.UNCLEAR
    return reading, None


def _decl(
    declarations: Mapping[str, ValueDeclaration], name: str
) -> ValueDeclaration:
    decl = declarations.get(name)
    if decl is None:
        raise CriteriaEvaluationError(
            f"value {name!r} is not declared (import-time validation "
            f"would have rejected this rule; declarations: "
            f"{sorted(declarations) or 'none'})"
        )
    return decl


def _number(decl: ValueDeclaration, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CriteriaEvaluationError(
            f"value {decl.name!r} reading {value!r} is not a number "
            f"(declared type: {decl.type})"
        )
    return float(value)


def _status_of(condition: bool) -> CriterionStatus:
    return CriterionStatus.SATISFIED if condition else CriterionStatus.NOT_SATISFIED


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _leaf_single_value(
    rule: CriteriaRule,
    readings: Mapping[str, ValueReading],
    declarations: Mapping[str, ValueDeclaration],
) -> CriterionStatus:
    """
    The single-value operators (eq / neq / gte / lte / between / min /
    max / threshold / contains / not_contains / is_empty /
    is_not_empty). Every leaf starts from the same four-valued reading
    verdict (missing -> OPEN, unreliable -> UNCLEAR, type mismatch ->
    UNCLEAR) and only then computes the boolean condition.
    """
    name = rule.value
    reading, verdict = _reading(name, readings)
    if verdict is not None:
        return verdict
    decl = _decl(declarations, name)
    if not _declared_type_ok(decl, reading.value):
        return CriterionStatus.UNCLEAR
    if reading.value is None:
        # An "empty" reading: the answer carried no extractable content.
        if isinstance(rule, IsEmptyRule):
            return CriterionStatus.SATISFIED
        if isinstance(rule, IsNotEmptyRule):
            return CriterionStatus.NOT_SATISFIED
        if isinstance(rule, ContainsRule):
            return CriterionStatus.NOT_SATISFIED  # nothing in an empty answer
        if isinstance(rule, (NotContainsRule, NeqRule)):
            return CriterionStatus.SATISFIED
        if isinstance(rule, EqRule):
            return CriterionStatus.NOT_SATISFIED  # never equals a literal
        # gte/lte/between/min/max/threshold: no number to compare ->
        # UNCLEAR (never NOT_SATISFIED for an uninterpretable answer)
        return CriterionStatus.UNCLEAR
    if isinstance(rule, (EqRule, NeqRule)):
        literal = rule.equals if isinstance(rule, EqRule) else rule.not_equals
        matches = reading.value == literal
        return _status_of(matches if isinstance(rule, EqRule) else not matches)
    if isinstance(rule, (GteRule, LteRule, MinRule, MaxRule, BetweenRule)):
        number = _number(decl, reading.value)
        if isinstance(rule, (GteRule, MinRule)):
            return _status_of(number >= rule.threshold)
        if isinstance(rule, (LteRule, MaxRule)):
            return _status_of(number <= rule.threshold)
        return _status_of(rule.min <= number <= rule.max)  # BetweenRule
    if isinstance(rule, ThresholdRule):
        named = decl.thresholds.get(rule.threshold)
        if named is None:
            raise CriteriaEvaluationError(
                f"threshold {rule.threshold!r} is not declared on value "
                f"{name!r} (declared thresholds: "
                f"{sorted(decl.thresholds) or 'none'})"
            )
        return _status_of(_number(decl, reading.value) >= named)
    if isinstance(rule, (ContainsRule, NotContainsRule)):
        needle = rule.needle.lower()
        found = needle in str(reading.value).lower()
        return _status_of(found if isinstance(rule, ContainsRule) else not found)
    empty = _is_empty(reading.value)
    return _status_of(empty if isinstance(rule, IsEmptyRule) else not empty)


def _leaf_aggregation(
    rule: CountRule | MajorityRule | WeightedSumRule,
    readings: Mapping[str, ValueReading],
    declarations: Mapping[str, ValueDeclaration],
) -> CriterionStatus:
    """
    The group-level aggregation operators (count / majority /
    weighted_sum). The rule refers to ALL the listed values together
    (the criteria group's elements): one missing value -> OPEN, one
    unreliable value -> UNCLEAR, one type-mismatched value -> UNCLEAR;
    only a fully evaluable group is counted/summed.
    """
    verdicts = [_reading(name, readings)[1] for name in rule.values]
    if any(v is CriterionStatus.UNCLEAR for v in verdicts):
        return CriterionStatus.UNCLEAR
    if any(v is CriterionStatus.OPEN for v in verdicts):
        return CriterionStatus.OPEN
    if any(readings[name].value is None for name in rule.values):
        return CriterionStatus.UNCLEAR  # an empty element cannot be counted/summed
    for name in rule.values:
        reading = readings[name]
        decl = _decl(declarations, name)
        if not _declared_type_ok(decl, reading.value):
            return CriterionStatus.UNCLEAR
    if isinstance(rule, WeightedSumRule):
        total = 0.0
        for name in rule.values:
            reading = readings[name]
            decl = _decl(declarations, name)
            weight = rule.weights.get(name)
            if weight is None:
                raise CriteriaEvaluationError(
                    f"weighted_sum has no weight for value {name!r} "
                    f"(import-time validation would have rejected this "
                    f"rule; weights: {sorted(rule.weights) or 'none'})"
                )
            total += weight * _number(decl, reading.value)
        return _status_of(total >= rule.threshold)
    count = sum(
        1
        for name in rule.values
        if readings[name].value == rule.if_eq
    )
    if isinstance(rule, CountRule):
        return _status_of(count >= rule.gte)
    return _status_of(count > len(rule.values) / 2)  # MajorityRule


def _evaluate_leaf(
    rule: CriteriaRule,
    readings: Mapping[str, ValueReading],
    declarations: Mapping[str, ValueDeclaration],
) -> CriterionStatus:
    """Dispatch one non-logical rule to its evaluator."""
    if isinstance(rule, (CountRule, MajorityRule, WeightedSumRule)):
        return _leaf_aggregation(rule, readings, declarations)
    return _leaf_single_value(rule, readings, declarations)


def _combine(children: list[CriterionStatus], op: str) -> CriterionStatus:
    """The four-valued combination of the logical operators (module docstring)."""
    if op == "not":
        (child,) = children
        if child is CriterionStatus.SATISFIED:
            return CriterionStatus.NOT_SATISFIED
        if child is CriterionStatus.NOT_SATISFIED:
            return CriterionStatus.SATISFIED
        return child  # OPEN stays OPEN, UNCLEAR stays UNCLEAR
    if op == "all":
        if any(c is CriterionStatus.NOT_SATISFIED for c in children):
            return CriterionStatus.NOT_SATISFIED
        if all(c is CriterionStatus.SATISFIED for c in children):
            return CriterionStatus.SATISFIED
        if any(c is CriterionStatus.UNCLEAR for c in children):
            return CriterionStatus.UNCLEAR
        return CriterionStatus.OPEN
    if op == "any":
        if any(c is CriterionStatus.SATISFIED for c in children):
            return CriterionStatus.SATISFIED
        if all(c is CriterionStatus.NOT_SATISFIED for c in children):
            return CriterionStatus.NOT_SATISFIED
        if any(c is CriterionStatus.UNCLEAR for c in children):
            return CriterionStatus.UNCLEAR
        return CriterionStatus.OPEN
    raise CriteriaEvaluationError(f"unknown logical operator {op!r}")  # pragma: no cover


def evaluate_rule(
    rule: CriteriaRule,
    readings: Mapping[str, ValueReading],
    declarations: Mapping[str, ValueDeclaration],
) -> CriterionStatus:
    """
    Evaluate one rule AST against the current value readings.

    readings: value-name -> ValueReading (absent name = not answered
    yet -> OPEN; reliable=False -> UNCLEAR). declarations: the
    definition's value declarations (type checks, named thresholds).
    """
    if isinstance(rule, (AllRule, AnyRule)):
        return _combine(
            [
                evaluate_rule(sub, readings, declarations)
                for sub in rule.rules
            ],
            rule.op,
        )
    if isinstance(rule, NotRule):
        return _combine(
            [evaluate_rule(rule.rule, readings, declarations)], "not"
        )
    return _evaluate_leaf(rule, readings, declarations)


def evaluate_criterion(
    criterion: CriterionDefinition,
    readings: Mapping[str, ValueReading],
    declarations: Mapping[str, ValueDeclaration],
) -> CriterionStatus:
    """Convenience: evaluate one declared criterion's rule."""
    return evaluate_rule(criterion.rule, readings, declarations)


def readings_for_criterion(
    criterion: CriterionDefinition,
    readings_by_context: Mapping[str, Mapping[str, ValueReading]],
    active_context: str = "now",
    active_is_past_episode: bool = False,
) -> Mapping[str, ValueReading]:
    """
    Pick the readings a criterion is evaluated against. 
    `readings_by_context` maps an episode id ("now",
    "ep_1", ...) to that episode's current readings:

      * run-scoped (no scope) — the readings of EVERY context are
        merged (later context overrides earlier — the legacy behavior
        for a single merged mapping);
      * scope `current` — only the ACTIVE context's readings (the
        now-episode at first, a declared past episode after a re-entry);
      * scope `past` — the active context's readings when the active
        episode is a PAST episode; in the now-episode context the
        criterion can never be decided (its values are not asked there),
        so the selection is EMPTY there (which evaluates to OPEN).

    This is what keeps a past episode's values from contaminating the
    now-episode's criterion decisions (and vice versa).
    """
    scope = criterion.scope
    if scope is None:
        merged: dict[str, ValueReading] = {}
        for context_readings in readings_by_context.values():
            merged.update(context_readings)
        return merged
    if scope.episode == "past" and not active_is_past_episode:
        return {}
    return dict(readings_by_context.get(active_context, {}))


def evaluate_criteria(
    definition: InterviewDefinition,
    readings: Mapping[str, ValueReading],
) -> dict[str, CriterionStatus]:
    """
    Evaluate ALL criteria of the definition against the readings —
    criterion id -> status. The result for a criterion whose values are
    not (fully) present yet is OPEN/UNCLEAR; that is what makes calling
    this after EVERY answer safe (the online character of the design
    note): the criteria whose answers are complete evaluate to a
    decided status, the rest stay open.
    """
    declarations = {v.name: v for v in definition.values}
    return {
        c.id: evaluate_criterion(c, readings, declarations)
        for c in definition.all_criteria()
    }