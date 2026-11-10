"""
INTERVIEW NAVIGATOR — THE DETERMINISTIC NAVIGATION TOOL (PHASE 2, 2.4)
=======================================================================
The "Interview Navi" (docs/interview_navi_design.md — the binding design
reference): a DETERMINISTIC TOOL (not an LLM agent) that is the
authoritative source for the interview's navigation. It does not
diagnose, formulate participant-facing content, or mutate anything: it
computes navigation decisions over (InterviewDefinition, InterviewRun,
value readings) and returns them as proposals. The Supervisor (step
2.5) is the sole authority that applies them to the run (the only
mutation path, step 2.2) and writes the audit trail.

Responsibilities (design note, "Bezug zum Interviewer-Agenten"):
  * maintain the "questions to be asked" set — the still-open questions
    of the run (outstanding_questions / open_questions),
  * hand the interviewer agent a CANDIDATE SET (contracts.
    QuestionCandidateSet) to pick the most sensible next question
    from — ordered by the deterministic fallback order with a per-
    candidate reason,
  * the DETERMINISTIC FALLBACK ORDER — the postorder traversal of the
    hierarchy: nested sections first, then question groups in listed
    order, then questions in their defined order; when no criteria
    decide otherwise, that order IS the question order,
  * resolve JUMPS after a question's answer is recorded and
    interpreted: a jump fires when the gated criterion's state in the
    run matches the jump's `when` (a jump without on_criterion fires
    unconditionally); the FIRST declared jump that fires wins; after a
    jump, navigation follows the postorder traversal from the target;
    questions bypassed by the jump receive SKIPPED proposals
    ("not needed in the current flow"); a question is never re-asked
    in the same run (a jump target that is fully answered — "locked"
    — is handled by the loop policy below),
  * evaluate criteria groups ONLINE (design note: criteria are
    evaluated continuously, as soon as the answers that feed them are
    complete): propose_criterion_updates evaluates every criterion
    after each answer — the criteria whose values are present evaluate
    to a decided status, the rest stay OPEN/UNCLEAR — and applies the
    DECIDED GUARD: a criterion already SATISFIED/NOT_SATISFIED is not
    re-evaluated (reopening only through UNCLEAR, step 2.2's
    criterion transition table),
   * (WInf-inspired) the DETERMINISTIC STEERING REACTIONS: the
     explicit-skip pre-check (is_explicit_skip — normalized exact
     match, no LLM) and the steering reaction table
     (steering_decision — intent + answer clarity + the per-kind
     re-ask counters + the SteeringConfig thresholds -> PROCEED /
     REASK (clarify / redirect / gentle / rephrase) / SKIP (explicit
     / refused / unresolved)); a skipped question never counts as a
     negative answer (no readings, the criteria stay OPEN/UNCLEAR),
     and a technical (model) failure is never a participant action
     (the Supervisor's failure policy with the PAUSED run state).

LOOP-BACK to an already-answered (LOCKED) target (RECONSTRUCTION_PLAN.md
§5, open question 6 — resolved by the episode model, design addendum
2026-10-26 + 2026-10-31): a jump whose target subtree has no open
questions (no more question to ask there).
  * a PLAIN jump (no episode aspect) is decided by the LoopPolicy —
    IGNORE_AND_CONTINUE (DEFAULT, safe): the jump is ignored and the
    deterministic postorder continues from the current position;
    END_RUN: everything remaining is skipped and the next question is
    None; ERROR: NavigatorError. Every locked-target decision carries
    a loop_note, so the decision is LOGGED, never silent.
  * an EPISODE jump re-asks the locked target FOR THE OTHER EPISODE
    (the re-entry, design addendum 2026-10-31, OFQ-1):
      * predeclared (`episode: <name>`) — deterministic: the named
        (canonical) episode decides; the target is re-asked for it
        unless it was already assessed for it (then ignored + logged);
      * dynamic (`episode_jump: true`) — PENDING until the
        Diagnostiker's EpisodeDecision (new_episode -> re-ask the
        target for the declared episode; same_episode -> the target
        was already assessed there, ignored + logged).
JUMPS ARE DEFERRED (owner decision 2026-10-26): a jump is only
RESOLVED once its question's group is complete (group_is_complete /
complete_group); a mid-group resolve_jump fails loudly.

OUT-OF-SCOPE EXIT (design addendum 2026-10-31, OFQ-6): a fired jump
marked `out_of_scope` is NOT navigated — the resolution carries
out_of_scope=True and the Supervisor aborts the run with
RunAbortCause.OUT_OF_SCOPE (the routed continuation target lies
outside the loaded definition, e.g. SCID A54 -> A78).

Used by: the Supervisor (step 2.5) and the tests (step 2.6). The
legacy pipeline does not use this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Mapping, Sequence

import re

from contracts import (
    AnswerClarity,
    CriterionStatus,
    InputIntent,
    QuestionCandidate,
    QuestionCandidateSet,
    QuestionState,
    RunStatus,
    SteeringConfig,
    ValueReading,
)
from criteria_engine import evaluate_criterion, readings_for_criterion
from interview_definition import (
    InterviewDefinition,
    Jump,
    QuestionGroup,
    SectionGroup,
    TEXT_TOKEN_RE,
)
from interview_run import InterviewRun


class LoopPolicy(Enum):
    """
    What the Navigator does when a jump leads to an already-answered
    (locked) group (see the module docstring). The default
    IGNORE_AND_CONTINUE is a pragmatic choice pending the open decision
    (RECONSTRUCTION_PLAN.md §5, question 6).
    """

    IGNORE_AND_CONTINUE = "ignore_and_continue"
    END_RUN = "end_run"
    ERROR = "error"


class NavigatorError(RuntimeError):
    """
    A navigation situation the Navigator cannot resolve (the ERROR loop
    policy, or an unknown jump target). Fails loudly instead of
    guessing.
    """


@dataclass(frozen=True)
class EpisodeDecision:
    """
    The Diagnostiker's decision about a PENDING dynamic episode jump
    (design addendum 2026-10-31, OFQ-1): which episode the jump's
    continuation belongs to, and whether the target group was ALREADY
    assessed for that (canonical) episode:
      * new_episode    — the target is re-asked FOR that episode
        (re-entry into the locked target);
      * same_episode   — the declaration refers to an episode the
        target was already assessed for: the jump is ignored + logged
        (the same question is never re-asked for the same real
        episode), the postorder continues.
    `episode` is the declared episode id (the Navigator resolves it to
    its canonical id; an unknown id fails loudly).
    """

    NEW_EPISODE: ClassVar[str] = "new_episode"
    SAME_EPISODE: ClassVar[str] = "same_episode"

    episode: str
    decision: str


@dataclass(frozen=True)
class JumpResolution:
    """
    The outcome of resolving the jumps declared on one answered
    question. A PROPOSAL — the Supervisor applies it:
      * skip_question() for every id in skip_proposals (UNSEEN ->
        REQUIRED -> SKIPPED; "not needed in the current flow"),
      * reopen_question(qid, episode) for every id in reopen_proposals
        (CLOSED/SKIPPED -> REQUIRED for the episode — the re-entry),
      * then require_question()/reserve_question() for next_question
        (when not None),
      * when policy_applied is END_RUN and next_question is None:
        skip the proposals and complete_run(),
      * when out_of_scope: abort_out_of_scope(target_id, jump.note)
        (OFQ-6) — the continuation target lies outside this
        definition; nothing else is applied.
    The EPISODE-MODEL fields (design addendum 2026-10-31):
      * pending_episode — a dynamic episode jump is PENDING: no
        decision until the Diagnostiker supplies an EpisodeDecision
        (a re-called resolve_jump with the decision applies it);
      * episode         — the (canonical) episode the resolution
        concerns (re-entry or same-episode ignore); None otherwise;
      * reopen_proposals — the target questions to re-ask for the
        episode (in postorder; the first is next_question);
      * reframe_context — the framing the Sprecher must carry into the
        re-ask ("... for the episode 'ep_1' (march 2026), which was
        previously assessed for the now-episode");
      * loop_note       — the logged decision for a LOCKED target
        (plain loop ignored + logged / same-episode ignore /
        out-of-scope exit) — never silent.
    """

    fired: bool
    jump: Jump | None
    target_id: str | None
    next_question: str | None
    skip_proposals: tuple[str, ...]
    locked_target: bool
    policy_applied: LoopPolicy | None
    # the episode model (design addendum 2026-10-31, OFQ-1):
    pending_episode: bool = False
    episode: str | None = None
    reopen_proposals: tuple[str, ...] = ()
    reframe_context: str | None = None
    loop_note: str | None = None
    # the out-of-scope exit (OFQ-6): the Supervisor aborts the run.
    out_of_scope: bool = False


@dataclass(frozen=True)
class GroupCompletion:
    """
    The consequence of a question group becoming COMPLETE (owner
    decision 2026-10-26 — deferred jumps): the group's jumps are
    resolved ONCE, at completion time (reading the criterion states as
    they are then), and the still-open OPTIONAL questions of the group
    become skip proposals (they are not needed once the group is done).
    A PROPOSAL — the Supervisor applies it.
    """

    group_id: str
    resolution: JumpResolution
    optional_skip_proposals: tuple[str, ...]


# ---------------------------------------------------------------------------
# STEERING — THE DETERMINISTIC STEERING REACTIONS (WInf-inspired)
# ---------------------------------------------------------------------------
# (WInf-inspired) the steering policy of the reference implementation:
# one input is classified by the Interventionsmonitor (contracts.
# InputIntent), and the reaction is DETERMINISTIC — a function of
# (intent, answer clarity, the per-kind re-ask counters of the current
# question, the SteeringConfig thresholds). No LLM is involved in the
# reaction itself; the LLM (Diagnostiker, Phase 3) only supplies the
# clarity of a plain answer, and the monitor (Phase 4) the intent.
#
# WInf-inspired invariants:
#   * a technical (model) failure is NEVER a participant action —
#     handled by the Supervisor (the pause policy with RunStatus.PAUSED);
#   * a skipped question NEVER counts as a negative answer: no readings
#     are recorded, the criteria depending on it stay OPEN/UNCLEAR;
#   * a SKIP REQUEST (the explicit skip command / the refusal intent)
#     is NOT acted on immediately: the question is PROBED first (the
#     participant is asked WHY they want to skip and offered
#     alternatives — a vague/short answer is fine), and only skipped
#     once the skip-probe threshold (max_skip_probes) is reached;
#   * the per-kind budgets are PER QUESTION and PER KIND (one redirect
#     does not consume the gentle-follow-up budget);
#   * clarification requests (the participant asks for the question to
#     be rephrased) do NOT consume the clarification budget (the
#     rephrase re-ask is unlimited by default — WInf RT01).


class ReaskKind(Enum):
    """
    (WInf-inspired) the KIND of a steering re-ask of a REVISED
    question — the kind owns its own counter (the per-kind attempt
    counter) and its own threshold (SteeringConfig). Lowercase values:
    wire vocabulary (the kind appears in audit causes and, in Phase 3,
    in the Diagnostiker/Sprecher prompts).
    """

    # the answer was ambiguous/incomplete (Diagnostiker: UNCLEAR)
    CLARIFICATION = "clarify"
    # the answer was off-topic or unclear (intents off_topic/unclear)
    REDIRECT = "redirect"
    # the participant cannot answer (intent cannot_answer)
    GENTLE = "gentle"
    # the participant did not understand / asks again (intent
    # clarification_request) — unlimited by default (WInf RT01)
    REPRISE = "rephrase"
    # the SKIP PROBE (the explicit skip command / the refusal intent):
    # the participant wants the question skipped — the system ASKS WHY
    # and offers alternatives (a vague/short answer is fine) before it
    # skips (the question is probed, not skipped immediately)
    SKIP_PROBE = "skip_probe"


class SteeringAction(Enum):
    """
    (WInf-inspired) the deterministic steering action for one input:
      PROCEED — the answer is sufficient: record + close (the classic
                advance);
      REASK   — the question is REVISED and re-asked (FOLLOWUP), with
                the decision's reask_kind (the kind's counter is
                incremented, the budget is consumed);
      SKIP    — the question is skipped (SKIPPED, locked): the
                participant's skip request (explicit skip command /
                refusal) reached the skip-probe threshold, or the
                kind's budget is exhausted (unresolved).
    """

    PROCEED = "proceed"
    REASK = "reask"
    SKIP = "skip"


class SkipReason(Enum):
    """
    (WInf-inspired) WHY a question is skipped — the audit cause must be
    able to say which of the three WInf skip reasons applied.
    """

    # the participant explicitly asked to skip (the deterministic
    # skip-phrase pre-check — no LLM involved); the skip happens only
    # once the skip-probe threshold is reached (the question is probed
    # first: ask why + alternatives)
    EXPLICIT = "explicit"
    # the participant refused to answer (intent refusal); like EXPLICIT,
    # the refusal is probed first (ask why + alternatives) and the skip
    # happens only once the skip-probe threshold is reached
    REFUSED = "refused"
    # the per-kind re-ask budget was exhausted (the item stays
    # unresolved) — recorded as unresolved, never as a negative answer
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class SteeringDecision:
    """
    (WInf-inspired) the outcome of the deterministic steering reaction
    table for one input — a PROPOSAL the Supervisor applies:
      * PROCEED — the classic record + close + advance;
      * REASK   — apply_answer(closed=False) + revise_question (the
                  re-ask re-enters via REVISED -> DELIVERED) + deliver
                  the FOLLOWUP; reask_kind names the budget consumed;
      * SKIP    — skip_question (DELIVERED -> SKIPPED) + advance;
                  skip_reason names the WInf skip reason.
    `reason` is a human-readable (auditable) justification.
    """

    action: SteeringAction
    reask_kind: ReaskKind | None = None
    skip_reason: SkipReason | None = None
    reason: str = ""


# (WInf-inspired) the politeness fillers the command normalization
# ignores, so "bitte überspringen" == "überspringen" and
# "please skip" == "skip":
_FILLER_RE = re.compile(r"\b(bitte|please|gerne|einfach)\b")
_MULTI_WORD_FILLER_RE = re.compile(r"\b(ich möchte|i want to)\b")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_command(text: str) -> str:
    """
    (WInf-inspired) normalize ONE participant input for the
    deterministic command checks (the explicit skip command, the
    failure-choice continue/pause commands): lowercase, everything
    non-alphanumeric collapsed to whitespace, the politeness fillers
    dropped, whitespace squeezed. The checks themselves are EXACT
    matches on this normalized form — deliberately strict, so a normal
    answer that merely contains a command word is never treated as the
    command.
    """
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    cleaned = _MULTI_WORD_FILLER_RE.sub(" ", cleaned)
    cleaned = _FILLER_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def is_explicit_skip(text: str, skip_phrases: Sequence[str]) -> bool:
    """
    (WInf-inspired) the DETERMINISTIC explicit-skip pre-check: is this
    input an explicit skip command for the CURRENT question?
    Normalized exact match against the configured skip phrases (no LLM,
    no fuzzy matching — the WInf skip check is intentionally strict, so
    a normal answer is never treated as a skip). Runs BEFORE the
    Diagnostiker in the pipeline — the skip decision needs no LLM.
    """
    if not (text or "").strip():
        return False
    normalized = normalize_command(text)
    if not normalized:
        return False
    return any(
        normalized == normalize_command(phrase)
        for phrase in skip_phrases
        if (phrase or "").strip()
    )


def _reask_budget_left(
    kind: ReaskKind, count: int, steering: SteeringConfig
) -> bool:
    """Whether the per-kind budget of ONE question still allows a re-ask."""
    if kind is ReaskKind.CLARIFICATION:
        limit = steering.max_clarifications
    elif kind is ReaskKind.REDIRECT:
        limit = steering.max_redirects
    elif kind is ReaskKind.GENTLE:
        limit = steering.max_gentle_followups
    elif kind is ReaskKind.SKIP_PROBE:
        limit = steering.max_skip_probes
    else:  # REPRISE
        limit = steering.max_rephrasings
    # None = unlimited (the rephrase re-ask, WInf RT01)
    return limit is None or count < limit


def skip_request_decision(
    reask_counts: Mapping[str, int],
    steering: SteeringConfig,
    skip_reason: SkipReason,
) -> SteeringDecision:
    """
    (WInf-inspired) the reaction to a SKIP REQUEST — the explicit skip
    command (the deterministic command check, skip_reason=EXPLICIT) or
    the refusal intent (the monitor, skip_reason=REFUSED). The
    question is NOT skipped immediately: while the skip-probe budget
    (SteeringConfig.max_skip_probes, per question) still allows, the
    system ASKS the participant WHY they want to skip and offers
    alternatives (a vague/short answer is fine — the pre-approved
    probe text) — the question is re-asked as a FOLLOWUP with that
    probe (the kind skip_probe owns its counter and threshold). Only
    once the probe threshold is reached the question is actually
    SKIPPED (with `skip_reason`; a skip is never a negative answer).
    """
    kind = ReaskKind.SKIP_PROBE
    if _reask_budget_left(kind, reask_counts.get(kind.value, 0), steering):
        return SteeringDecision(
            action=SteeringAction.REASK,
            reask_kind=kind,
            reason=(
                "the participant wants the question skipped — it is "
                "probed first (asked why + alternatives offered: a "
                "vague/short answer is fine) before a skip"
            ),
        )
    return SteeringDecision(
        action=SteeringAction.SKIP,
        skip_reason=skip_reason,
        reason=(
            f"the skip-probe threshold (max_skip_probes="
            f"{steering.max_skip_probes}) is reached — the question is "
            f"skipped ({skip_reason.value})"
        ),
    )


def steering_decision(
    intent: InputIntent,
    clarity: AnswerClarity,
    reask_counts: Mapping[str, int],
    steering: SteeringConfig,
) -> SteeringDecision:
    """
    (WInf-inspired) the DETERMINISTIC steering reaction table — a pure
    function of (intent, clarity, the current question's per-kind
    re-ask counters, the SteeringConfig):

      intent REFUSAL               -> the SKIP-REQUEST reaction
                                        (skip_request_decision — the
                                        question is probed first: ask
                                        why + alternatives; the skip
                                        happens only once the
                                        skip-probe threshold is reached)
      intent CLARIFICATION_REQUEST -> REASK (rephrase)
                                        [WInf RT01: unlimited by DEFAULT
                                         (max_rephrasings=None) — does
                                         NOT consume the clarification
                                         budget; a finite configured
                                         threshold still gates it]
      intent OFF_TOPIC / UNCLEAR   -> REASK (redirect) while the budget
                                      allows, else SKIP (unresolved)
      intent CANNOT_ANSWER         -> REASK (gentle) while the budget
                                      allows, else SKIP (unresolved)
      intent ANSWER:
        clarity CLEAR              -> PROCEED
        clarity UNCLEAR            -> REASK (clarify) while the budget
                                      allows, else SKIP (unresolved)

    The explicit skip command is NOT part of this table either: the
    Supervisor checks it BEFORE the Diagnostiker (is_explicit_skip —
    no LLM involved, so it does not even need the clarity) and applies
    the SAME skip_request_decision with SkipReason.EXPLICIT.
    `reask_counts` maps the kind's wire value (ReaskKind.value) to the
    number of re-asks of that kind already given for this question.
    """
    if intent is InputIntent.REFUSAL:
        return skip_request_decision(reask_counts, steering, SkipReason.REFUSED)
    if intent is InputIntent.CLARIFICATION_REQUEST:
        kind = ReaskKind.REPRISE
        if _reask_budget_left(kind, reask_counts.get(kind.value, 0), steering):
            return SteeringDecision(
                action=SteeringAction.REASK,
                reask_kind=kind,
                reason=(
                    "the participant asked to have the question "
                    "rephrased (intent: clarification_request — "
                    "unlimited by default, WInf RT01)"
                ),
            )
        return SteeringDecision(
            action=SteeringAction.SKIP,
            skip_reason=SkipReason.UNRESOLVED,
            reason=(
                "the rephrase budget is exhausted (intent: "
                "clarification_request) — the question is skipped as "
                "unresolved (never a negative answer)"
            ),
        )
    if intent in (InputIntent.OFF_TOPIC, InputIntent.UNCLEAR):
        kind = ReaskKind.REDIRECT
        if _reask_budget_left(kind, reask_counts.get(kind.value, 0), steering):
            return SteeringDecision(
                action=SteeringAction.REASK,
                reask_kind=kind,
                reason=(
                    "the answer was off-topic/unclear (intent: "
                    f"{intent.value}) — a redirect before skip"
                ),
            )
        return SteeringDecision(
            action=SteeringAction.SKIP,
            skip_reason=SkipReason.UNRESOLVED,
            reason=(
                "the redirect budget is exhausted (intent: "
                f"{intent.value}) — the question is skipped as "
                "unresolved (never a negative answer)"
            ),
        )
    if intent is InputIntent.CANNOT_ANSWER:
        kind = ReaskKind.GENTLE
        if _reask_budget_left(kind, reask_counts.get(kind.value, 0), steering):
            return SteeringDecision(
                action=SteeringAction.REASK,
                reask_kind=kind,
                reason=(
                    "the participant cannot answer (intent: "
                    "cannot_answer) — a gentle follow-up before skip"
                ),
            )
        return SteeringDecision(
            action=SteeringAction.SKIP,
            skip_reason=SkipReason.UNRESOLVED,
            reason=(
                "the gentle-follow-up budget is exhausted (intent: "
                "cannot_answer) — the question is skipped as "
                "unresolved (never a negative answer)"
            ),
        )
    # intent ANSWER: the classic clarity-only path
    if clarity is AnswerClarity.CLEAR:
        return SteeringDecision(
            action=SteeringAction.PROCEED,
            reason="the answer is clear — the question is closed",
        )
    kind = ReaskKind.CLARIFICATION
    if _reask_budget_left(kind, reask_counts.get(kind.value, 0), steering):
        return SteeringDecision(
            action=SteeringAction.REASK,
            reask_kind=kind,
            reason=(
                "the answer lacks information — a clarification "
                "before skip"
            ),
        )
    return SteeringDecision(
        action=SteeringAction.SKIP,
        skip_reason=SkipReason.UNRESOLVED,
        reason=(
            "the clarification budget is exhausted — the question is "
            "skipped as unresolved (never a negative answer)"
        ),
    )


# The question states of the state machine (step 2.2) this tool reads:
_TERMINAL_STATES = frozenset({QuestionState.CLOSED, QuestionState.SKIPPED})  # answered or never-to-be-delivered
_CANDIDATE_STATES = frozenset({QuestionState.UNSEEN, QuestionState.REQUIRED})  # can be asked (not yet in flight, not yet decided)


class InterviewNavigator:
    """
    The deterministic navigation tool for ONE interview definition.

    Pure and stateless per call: every method is a function of the
    (definition, run, readings) it is given — the same inputs always
    give the same proposals. The navigator never mutates the run; the
    Supervisor applies the proposals through the state machine's
    transition methods (the only mutation path).
    """

    def __init__(
        self,
        definition: InterviewDefinition,
        loop_policy: LoopPolicy = LoopPolicy.IGNORE_AND_CONTINUE,
    ) -> None:
        self._definition = definition
        self._loop_policy = loop_policy
        # the deterministic fallback order: the postorder traversal of
        # the hierarchy — nested sections first, then question groups
        # in listed order, then questions in their defined order.
        self._order: list[str] = [q.id for q in definition.all_questions()]
        self._position: dict[str, int] = {
            qid: i for i, qid in enumerate(self._order)
        }
        self._declarations = {v.name: v for v in definition.values}
        self._question_group_of: dict[str, QuestionGroup] = {
            q.id: group for group, q in definition.questions_with_group()
        }

    # ------------------------------------------------------------------
    # structure access (the authoritative source for interview structure)
    # ------------------------------------------------------------------

    @property
    def definition(self) -> InterviewDefinition:
        return self._definition

    @property
    def loop_policy(self) -> LoopPolicy:
        return self._loop_policy

    @property
    def question_order(self) -> list[str]:
        """
        The deterministic fallback order (postorder / document order)
        of all single questions — QuestionCandidateSet.fallback_order.
        """
        return list(self._order)

    def _section_questions(self, section: SectionGroup) -> list[str]:
        """All questions in a section group's subtree, document order."""
        out: list[str] = []
        for child in section.children:
            if isinstance(child, SectionGroup):
                out.extend(self._section_questions(child))
            else:
                out.extend(q.id for q in child.questions)
        return out

    def _questions_of_target(self, target_id: str) -> list[str]:
        """
        The questions of a jump target's subtree (document order). The
        target is a question, question group or section group (import
        validation guarantees it exists — fail loudly if not).
        """
        question = self._definition.question_by_id(target_id)
        if question is not None:
            return [target_id]
        group = self._definition.group_by_id(target_id)
        if group is not None:
            return [q.id for q in group.questions]
        section = self._definition.section_by_id(target_id)
        if section is not None:
            return self._section_questions(section)
        raise NavigatorError(
            f"jump target {target_id!r} is not a known question, group or "
            f"section id (import-time validation would have rejected it)"
        )

    # ------------------------------------------------------------------
    # the "questions to be asked" set 
    # ------------------------------------------------------------------

    def outstanding_questions(self, run: InterviewRun) -> list[str]:
        """
        The still-open questions — everything that is not CLOSED or
        SKIPPED yet (UNSEEN .. REVISED), in postorder. The run ends
        only when this set is empty (is_complete).
        """
        states = run.question_states
        return [q for q in self._order if states[q] not in _TERMINAL_STATES]

    def open_questions(self, run: InterviewRun) -> list[str]:
        """
        The questions that can be ASKED right now — UNSEEN or REQUIRED
        (not yet in flight: RESERVED / DELIVERED / ANSWER_RECORDED /
        REVISED), in postorder.
        """
        states = run.question_states
        return [q for q in self._order if states[q] in _CANDIDATE_STATES]

    def is_complete(self, run: InterviewRun) -> bool:
        """
        Nothing more is to be asked: every question is CLOSED or
        SKIPPED. The Supervisor may then complete_run() (the ONLY way
        a run ends, step 2.2).
        """
        states = run.question_states
        return all(states[q] in _TERMINAL_STATES for q in self._order)

    # ------------------------------------------------------------------
    # the candidate set handed to the interviewer agent
    # ------------------------------------------------------------------

    def _candidate_reason(
        self, question_id: str, run: InterviewRun | None = None
    ) -> str:
        """
        The human-readable (auditable) reason for one candidate —
        including the EPISODE the question is (again) asked about
        (design addendum 2026-10-31): after a re-entry the re-asked
        question's reason names the past episode, so the agent never
        re-frames a past-episode question as the now-episode's.
        """
        question = self._definition.question_by_id(question_id)
        group = self._question_group_of[question_id]
        position = self._position[question_id] + 1
        required = "required" if question.required else "optional"
        episode = "now"
        if run is not None:
            asked = run.question_episodes.get(question_id)
            if asked:
                episode = asked[-1]
        return (
            f"postorder position {position} in group {group.id} "
            f"({group.strategy}), {required}, episode {episode}"
        )

    def _candidate_list(self, run: InterviewRun) -> list[QuestionCandidate]:
        """
        The candidates for the interviewer agent (the set's body).

        Rules (in priority order):
          * a terminal run (COMPLETED / ABORTED) has no candidates;
          * while a question is REVISED it is re-asked FIRST — the
            Navigator must not move on to a new question (the REVISED
            questions come first; a RESERVED question, if any,
            follows) — no new candidates;
          * while a question is RESERVED it is the single next
            question (the reservation was made, it is in flight);
          * while a question is DELIVERED or ANSWER_RECORDED the run
            waits for the answer — no candidates;
          * otherwise: every UNSEEN/REQUIRED question is a candidate,
            in the deterministic fallback order (postorder); ordering
            0 is the fallback next question. The interviewer agent may
            pick any candidate as "the most sensible next question";
            the Supervisor reserves the picked one.
        """
        states = run.question_states
        if run.status is not RunStatus.ACTIVE:
            return []
        revised = [q for q in self._order if states[q] is QuestionState.REVISED]
        reserved = run.reserved_question()
        if revised:
            return self._revised_candidates(revised, reserved, run)
        if reserved is not None:
            return self._reserved_candidates(reserved, run)
        in_flight = [
            q
            for q in self._order
            if states[q]
            in (QuestionState.DELIVERED, QuestionState.ANSWER_RECORDED)
        ]
        if in_flight:
            return []  # the answer is in flight; nothing to ask right now
        return self._open_candidates(run)

    def _revised_candidates(
        self,
        revised: list[str],
        reserved: str | None,
        run: InterviewRun,
    ) -> list[QuestionCandidate]:
        """REVISED questions are re-asked before any new question."""
        candidates = [
            QuestionCandidate(
                question_id=q,
                text=self.rendered_text(q, run),
                group_id=self._question_group_of[q].id,
                reason_required=(
                    "REVISED: re-ask before any new question "
                    f"({self._candidate_reason(q, run)})"
                ),
                ordering=i,
            )
            for i, q in enumerate(revised)
        ]
        if reserved is not None:
            candidates.extend(self._reserved_candidates(reserved, run))
        return candidates

    def _reserved_candidates(
        self, reserved: str, run: InterviewRun
    ) -> list[QuestionCandidate]:
        """The single RESERVED question, as the only candidate."""
        return [
            QuestionCandidate(
                question_id=reserved,
                text=self.rendered_text(reserved, run),
                group_id=self._question_group_of[reserved].id,
                reason_required=(
                    "RESERVED: the single next question "
                    f"({self._candidate_reason(reserved, run)})"
                ),
                ordering=0,
            )
        ]

    def _open_candidates(self, run: InterviewRun) -> list[QuestionCandidate]:
        """Every open (UNSEEN/REQUIRED) question, deterministic order."""
        return [
            QuestionCandidate(
                question_id=q,
                text=self.rendered_text(q, run),
                group_id=self._question_group_of[q].id,
                reason_required=self._candidate_reason(q, run),
                ordering=i,
            )
            for i, q in enumerate(self.open_questions(run))
        ]

    def rendered_text(self, question_id: str, run: InterviewRun) -> str:
        """
        The question's canonical text with its {text tokens} filled from
        the RUN's current state (design addendum 2026-10-31,
        OFQ-4/OFQ-5) — the wording stays SCID-faithful while adapting to
        the episode context:
          * {symptom_term} — the ACTIVE episode's symptom term: "euphorisch"
            when episode_elevated_mood holds, "reizbar" when
            episode_irritable_mood holds (default "euphorisch" before
            the mood is pinned — the SCID's euphoric framing);
          * {other_time} — the article for "another episode" (the A54
            screen): "eine andere" when the now-episode already has an
            established mood (so "another" is meaningful), else "eine".
        Tokens are a CLOSED set (validated at import — TEXT_TOKENS); a
        question without tokens is returned verbatim.
        """
        question = self._definition.question_by_id(question_id)
        if question is None:
            raise NavigatorError(f"unknown question id {question_id!r}")
        return self._render_text(question.text, run)

    def _render_text(self, text: str, run: InterviewRun) -> str:
        if "{" not in text:
            return text
        active = run.readings()  # the active episode context
        now_readings = run.readings("now")
        if self._is_true(active.get("episode_elevated_mood")):
            symptom_term = "euphorisch"
        elif self._is_true(active.get("episode_irritable_mood")):
            symptom_term = "reizbar"
        else:
            symptom_term = "euphorisch"
        now_mood_pinned = (
            self._is_true(now_readings.get("episode_elevated_mood"))
            or self._is_true(now_readings.get("episode_irritable_mood"))
        )
        values = {
            "symptom_term": symptom_term,
            "other_time": "eine andere" if now_mood_pinned else "eine",
        }
        return TEXT_TOKEN_RE.sub(
            lambda m: values.get(m.group(1), m.group(0)), text
        )

    @staticmethod
    def _is_true(reading: ValueReading | None) -> bool:
        """A reliable reading whose value is exactly True."""
        return (
            reading is not None
            and reading.reliable
            and reading.value is True
        )

    def candidate_set(self, run: InterviewRun, turn_id: str) -> QuestionCandidateSet:
        """
        The candidate set handed to the interviewer agent (contracts.
        QuestionCandidateSet, identified by run_id + turn_id like every
        other per-turn artifact): which questions may be asked next, in
        the deterministic fallback order, with a per-candidate reason.
        """
        return QuestionCandidateSet(
            run_id=run.run_id,
            turn_id=turn_id,
            candidates=self._candidate_list(run),
        )

    def next_question(self, run: InterviewRun) -> str | None:
        """
        The deterministic fallback next question (ordering 0 of the
        candidate list), or None when there is nothing to ask right
        now — either the run is complete (is_complete) or a question
        is in flight awaiting its answer (DELIVERED / ANSWER_RECORDED)
        or its re-ask.
        """
        candidates = self._candidate_list(run)
        if not candidates:
            return None
        return candidates[0].question_id

    # ------------------------------------------------------------------
    # jumps (transitions out of an answered question)
    # ------------------------------------------------------------------

    def _fallback_next(self, run: InterviewRun) -> str | None:
        """
        The fallback next question when a jump does not fire (or is
        ignored by the loop policy): the first open question in
        postorder. The source question of a resolve_jump call is still
        in flight (ANSWER_RECORDED) and is therefore NOT an open
        question — so this is exactly the question that next_question
        would return once the Supervisor has closed the source.
        """
        open_questions = self.open_questions(run)
        if not open_questions:
            return None
        return open_questions[0]

    def group_is_complete(self, run: InterviewRun, group_id: str) -> bool:
        """
        A question group is COMPLETE when every REQUIRED question of it
        has reached a terminal state (CLOSED or SKIPPED). Optional
        questions do not block completion (they are skip-proposed at
        completion). This is when the group's deferred jumps may be
        resolved (owner decision 2026-10-26). Unknown group ids fail
        loudly.
        """
        group = self._definition.group_by_id(group_id)
        if group is None:
            raise NavigatorError(
                f"unknown group id {group_id!r} (no such question group)"
            )
        states = run.question_states
        return all(
            states[q.id] in _TERMINAL_STATES
            for q in group.questions
            if q.required
        )

    def complete_group(
        self, run: InterviewRun, group_id: str
    ) -> GroupCompletion:
        """
        Resolve the DEFERRED jumps of a group that has just become
        complete (owner decision 2026-10-26): the (postorder) FIRST
        question of the group whose declared jump fires wins — the gate
        criterion is read at COMPLETION time (not when the group was
        entered). The still-open OPTIONAL questions of the group
        become skip proposals. A mid-group call (or an unknown group)
        fails loudly.
        """
        group = self._definition.group_by_id(group_id)
        if group is None:
            raise NavigatorError(
                f"unknown group id {group_id!r} (no such question group)"
            )
        if run.status is not RunStatus.ACTIVE:
            raise NavigatorError(
                f"the run is {run.status.value} — a group cannot be "
                f"completed on a terminal run"
            )
        if not self.group_is_complete(run, group_id):
            raise NavigatorError(
                f"group {group_id!r} is not complete yet — all required "
                f"questions must be terminal (CLOSED or SKIPPED)"
            )
        resolution: JumpResolution | None = None
        for question in group.questions:
            if not question.jumps:
                continue
            for jump in question.jumps:
                if self._jump_fires(run, jump):
                    resolution = self._resolve_fired_jump(
                        run, question.id, jump
                    )
                    break
            if resolution is not None:
                break
        if resolution is None:
            resolution = JumpResolution(
                fired=False,
                jump=None,
                target_id=None,
                next_question=self._fallback_next(run),
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
            )
        optional_skip_proposals = tuple(
            q.id
            for q in group.questions
            if not q.required
            and run.question_state(q.id) in _CANDIDATE_STATES
        )
        return GroupCompletion(
            group_id=group_id,
            resolution=resolution,
            optional_skip_proposals=optional_skip_proposals,
        )

    def _jump_fires(self, run: InterviewRun, jump: Jump) -> bool:
        """A jump fires when its gated criterion matches `when`."""
        if jump.on_criterion is None:
            return True
        return run.criterion_state(jump.on_criterion) is jump.when

    def resolve_jump(
        self,
        run: InterviewRun,
        question_id: str,
        episode_decision: EpisodeDecision | None = None,
    ) -> JumpResolution:
        """
        The completion-time primitive: resolve the jumps declared on
        `question_id` ONCE its group is complete (deferred jumps, owner
        decision 2026-10-26) — a mid-group call fails LOUDLY. A jump
        fires when the state of its gated criterion (on_criterion) in
        the run matches the jump's `when`; a jump without on_criterion
        fires unconditionally. The FIRST declared jump that fires wins
        (deterministic). When no jump fires, next_question is the
        plain deterministic fallback (postorder) and nothing is
        skipped.

        For a PENDING dynamic episode jump (pending_episode), the
        resolution is final only once the Diagnostiker's
        `episode_decision` is supplied (re-call with the decision).
        """
        question = self._definition.question_by_id(question_id)
        if question is None:
            raise NavigatorError(
                f"unknown question id {question_id!r} — nothing to resolve"
            )
        group = self._question_group_of[question_id]
        if not self.group_is_complete(run, group.id):
            raise NavigatorError(
                f"jump of {question_id!r} cannot be resolved mid-group: "
                f"group {group.id} is not complete yet (all required "
                f"questions must be terminal) — use complete_group() "
                f"once the group is worked through"
            )
        if run.status is not RunStatus.ACTIVE:
            return JumpResolution(
                fired=False,
                jump=None,
                target_id=None,
                next_question=None,
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
            )
        if not question.jumps:
            return JumpResolution(
                fired=False,
                jump=None,
                target_id=None,
                next_question=self._fallback_next(run),
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
            )
        for jump in question.jumps:
            if self._jump_fires(run, jump):
                return self._resolve_fired_jump(
                    run, question_id, jump, episode_decision
                )
        return JumpResolution(
            fired=False,
            jump=None,
            target_id=None,
            next_question=self._fallback_next(run),
            skip_proposals=(),
            locked_target=False,
            policy_applied=None,
        )

    def _resolve_fired_jump(
        self,
        run: InterviewRun,
        source_id: str,
        jump: Jump,
        episode_decision: EpisodeDecision | None = None,
    ) -> JumpResolution:
        """The outcome of one FIRED jump (target resolution + skips)."""
        # OUT-OF-SCOPE EXIT (OFQ-6): the continuation target lies
        # OUTSIDE this definition — an explicit, logged error; the
        # Supervisor aborts the run, nothing is navigated.
        if jump.out_of_scope:
            return JumpResolution(
                fired=True,
                jump=jump,
                target_id=jump.to,
                next_question=None,
                skip_proposals=(),
                locked_target=False,
                policy_applied=None,
                out_of_scope=True,
                loop_note=(
                    "out-of-scope exit: the interview's continuation "
                    f"target {jump.to!r} lies outside this definition "
                    f"(abort the run; {jump.note or 'no note given'})"
                ),
            )
        target_questions = self._questions_of_target(jump.to)
        states = run.question_states
        candidates = [
            q for q in target_questions if states[q] in _CANDIDATE_STATES
        ]
        if not candidates:
            # the target is LOCKED — every question of its subtree is
            # terminal (answered or skipped): the loop policy / the
            # episode model decide (see the module docstring).
            return self._resolve_locked_target(
                run, source_id, jump, episode_decision
            )
        next_question = candidates[0]
        source_pos = self._position[source_id]
        target_start = self._position[target_questions[0]]
        if target_start > source_pos:
            # FORWARD jump: the open questions strictly between the
            # source and the target subtree are bypassed by the jump
            # ("not needed in the current flow") -> SKIPPED proposals.
            skip_proposals = tuple(
                q
                for q in self.open_questions(run)
                if source_pos < self._position[q] < target_start
            )
        else:
            # BACKWARD jump: the flow re-enters the target subtree and
            # continues the postorder traversal from there — nothing
            # is bypassed permanently.
            skip_proposals = ()
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=next_question,
            skip_proposals=skip_proposals,
            locked_target=False,
            policy_applied=None,
        )

    def _apply_loop_policy(
        self, run: InterviewRun, source_id: str, jump: Jump
    ) -> JumpResolution:
        """
        A fired jump whose target is LOCKED (open question 6, §5).
        The locked questions are never re-asked under any policy.
        """
        if self._loop_policy is LoopPolicy.ERROR:
            raise NavigatorError(
                f"jump of {source_id!r} leads to the already answered "
                f"(locked) target {jump.to!r}; the loop-back behavior is "
                f"undecided (RECONSTRUCTION_PLAN.md, open question 6) "
                f"and the navigator is configured to fail loudly"
            )
        if self._loop_policy is LoopPolicy.END_RUN:
            # the loop counts as "nothing more to ask": everything
            # remaining is skipped, the run may be completed
            return JumpResolution(
                fired=True,
                jump=jump,
                target_id=jump.to,
                next_question=None,
                skip_proposals=tuple(self.open_questions(run)),
                locked_target=True,
                policy_applied=LoopPolicy.END_RUN,
            )
        # DEFAULT — IGNORE_AND_CONTINUE: the jump is ignored, the
        # deterministic postorder continues from the current position.
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=self._fallback_next(run),
            skip_proposals=(),
            locked_target=True,
            policy_applied=LoopPolicy.IGNORE_AND_CONTINUE,
        )

    def _resolve_locked_target(
        self,
        run: InterviewRun,
        source_id: str,
        jump: Jump,
        episode_decision: EpisodeDecision | None = None,
    ) -> JumpResolution:
        """
        A fired jump whose target is LOCKED (design addendum
        2026-10-26 + 2026-10-31, OFQ-1): the loop policy decides for a
        PLAIN jump (ignored + logged, never silent); an EPISODE jump
        re-asks the target FOR THE OTHER EPISODE (the re-entry) —
        deterministically for a predeclared name, via the
        Diagnostiker's EpisodeDecision for a dynamic one.
        """
        if not (jump.episode_jump or jump.episode is not None):
            # PLAIN jump: the decision is the loop policy's — and it
            # is LOGGED (the loop_note makes the decision auditable).
            base = self._apply_loop_policy(run, source_id, jump)
            tail = (
                " (postorder continues)"
                if base.policy_applied
                is LoopPolicy.IGNORE_AND_CONTINUE
                else " (everything remaining is skipped; the run may "
                "be completed)"
            )
            return JumpResolution(
                fired=base.fired,
                jump=jump,
                target_id=base.target_id,
                next_question=base.next_question,
                skip_proposals=base.skip_proposals,
                locked_target=True,
                policy_applied=base.policy_applied,
                loop_note=(
                    "jump to an already-answered target has no episode "
                    f"aspect — the loop is not re-navigated and is "
                    f"logged{tail}"
                ),
            )
        # EPISODE jump: the target is re-asked for the OTHER episode.
        if jump.episode is not None:
            # predeclared: the named (canonical) episode decides — no
            # runtime decision needed.
            episode = run.canonical_episode(jump.episode)
            if self.target_assessed_for(run, jump.to, episode):
                return self._episode_jump_ignored(
                    run,
                    jump,
                    episode,
                    f"episode jump ignored: the target was already "
                    f"assessed for episode '{episode}'"
                    f"{self._alias_ref(run, episode)} — the same "
                    f"question is not re-asked for the same real "
                    f"episode; postorder continues",
                )
            return self._episode_reask(run, jump, episode)
        # dynamic: the concrete episode is a RUNTIME outcome — PENDING
        # until the Diagnostiker's decision (OFQ-1).
        if episode_decision is None:
            return JumpResolution(
                fired=True,
                jump=jump,
                target_id=jump.to,
                next_question=None,
                skip_proposals=(),
                locked_target=True,
                policy_applied=None,
                pending_episode=True,
                loop_note=(
                    "episode jump pending: the Diagnostiker must decide "
                    "the EpisodeDecision for the locked target "
                    f"{jump.to!r} — new_episode (re-ask the target for "
                    "the declared episode) or same_episode (already "
                    "assessed; the loop is ignored)"
                ),
            )
        # the decision's episode is resolved to its canonical id —
        # an unknown id fails loudly (NavigatorError, not a plain
        # ValueError: this is a navigation situation, step 2.4).
        try:
            canonical = run.canonical_episode(episode_decision.episode)
        except ValueError as error:
            raise NavigatorError(
                f"episode decision references unknown episode "
                f"{episode_decision.episode!r}: {error}"
            ) from error
        if episode_decision.decision == EpisodeDecision.SAME_EPISODE:
            return self._episode_jump_ignored(
                run,
                jump,
                canonical,
                f"episode jump ignored: episode "
                f"'{episode_decision.episode}' was judged the SAME real "
                f"episode as '{canonical}'"
                f"{self._alias_ref(run, canonical)} — the target was "
                f"already assessed there; postorder continues",
            )
        if self.target_assessed_for(run, jump.to, canonical):
            # defensive: a new_episode decision for an episode the
            # target was already assessed for cannot re-ask it.
            return self._episode_jump_ignored(
                run,
                jump,
                canonical,
                f"episode jump ignored: the target was already "
                f"assessed for episode '{canonical}'"
                f"{self._alias_ref(run, canonical)} — the same "
                f"question is not re-asked for the same real episode; "
                f"postorder continues",
            )
        return self._episode_reask(run, jump, canonical)

    def _episode_reask(
        self, run: InterviewRun, jump: Jump, episode: str
    ) -> JumpResolution:
        """
        The RE-ENTRY (OFQ-1): re-ask the locked target FOR `episode`.
        Every terminal question of the target subtree becomes a reopen
        proposal (CLOSED/SKIPPED -> REQUIRED for the episode — the
        state machine's only move out of a terminal question state);
        the first (postorder) is the next question. The Supervisor
        applies the reopens, switches the active episode context
        (run.set_active_episode) and reserves the next question; the
        reframe_context names the new episode (with the patient's
        alias, when available) and the episode the target was
        previously assessed for.
        """
        target_questions = self._questions_of_target(jump.to)
        reopen = tuple(
            q
            for q in target_questions
            if run.question_state(q) in _TERMINAL_STATES
        )
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=reopen[0] if reopen else self._fallback_next(run),
            skip_proposals=(),
            locked_target=False,
            policy_applied=None,
            episode=episode,
            reopen_proposals=reopen,
            reframe_context=(
                f"Re-ask the locked target '{jump.to}' for the episode "
                f"'{episode}'{self._alias_ref(run, episode)}; it was "
                f"previously assessed for the now-episode — frame the "
                f"questions explicitly about this other episode."
            ),
        )

    def _episode_jump_ignored(
        self, run: InterviewRun, jump: Jump, episode: str, note: str
    ) -> JumpResolution:
        """An episode jump that must NOT re-ask the target (ignored + logged)."""
        return JumpResolution(
            fired=True,
            jump=jump,
            target_id=jump.to,
            next_question=self._fallback_next(run),
            skip_proposals=(),
            locked_target=True,
            policy_applied=None,
            episode=episode,
            loop_note=note,
        )

    def target_assessed_for(
        self, run: InterviewRun, target_ref: str, episode: str
    ) -> bool:
        """
        Whether the target subtree was already assessed for `episode`
        (any of its questions was asked about the canonical episode).
        """
        return any(
            episode in run.question_episodes.get(q, [])
            for q in self._questions_of_target(target_ref)
        )

    def _alias_ref(self, run: InterviewRun, episode: str) -> str:
        """The episode's first patient alias as a reference, if any."""
        episode_state = run.episodes.get(episode)
        if episode_state and episode_state.aliases:
            return f" (alias: '{episode_state.aliases[0]}')"
        return ""

    # ------------------------------------------------------------------
    # online criteria evaluation (the criteria groups of the design note)
    # ------------------------------------------------------------------

    def propose_criterion_updates(
        self,
        run: InterviewRun,
        readings: dict[str, ValueReading],
    ) -> dict[str, CriterionStatus]:
        """
        The ONLINE criteria evaluation (design note: criteria are
        evaluated continuously, as soon as the answers that feed them
        are complete). Every criterion is evaluated NOW — the engine
        returns OPEN/UNCLEAR for criteria whose values are not (fully)
        present yet — and exactly the status changes that differ from
        the current state are returned as proposals.

        The DECIDED GUARD (plan §1.1 rule 4): a criterion already
        SATISFIED or NOT_SATISFIED is NOT re-evaluated — its status
        only changes through UNCLEAR (reopened when a clarification
        shows the earlier answer was misread), never decided ->
        decided. That is exactly the state machine's criterion
        transition table (step 2.2); every proposal returned here is a
        legal transition (the current state is OPEN or UNCLEAR).

        The Supervisor applies the proposals via run.set_criterion()
        (e.g. cause "criteria engine after answer to sq_1.1.2") and
        writes them to the audit log.

        EPISODE SCOPING (design addendum 2026-10-31, OFQ-2): the
        `readings` belong to the ACTIVE episode context and are merged
        into that context's stored readings before evaluation; each
        criterion is then evaluated only against the readings its
        scope SELECTS (run-scoped: all contexts merged — the legacy
        behavior; `current`: the active context; `past`: a past
        context only) — so a past episode's values never contaminate
        the now-episode's decisions (and vice versa). The decided
        guard and the legal transitions are checked per context.
        """
        active = run.active_episode
        store = run.readings_by_context()
        active_readings = dict(store.get(active, {}))
        if readings:
            active_readings.update(readings)
        by_context = {**store, active: active_readings}
        active_is_past = run.episodes[active].kind == "past"
        proposed: dict[str, CriterionStatus] = {}
        for criterion in self._definition.all_criteria():
            current = run.criterion_state(criterion.id)
            if current in (
                CriterionStatus.SATISFIED,
                CriterionStatus.NOT_SATISFIED,
            ):
                continue  # decided guard (in the active context)
            visible = readings_for_criterion(
                criterion, by_context, active, active_is_past
            )
            new_status = evaluate_criterion(
                criterion, visible, self._declarations
            )
            if new_status is not current:
                proposed[criterion.id] = new_status
        return proposed