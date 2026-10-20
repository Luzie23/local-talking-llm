"""
STORE — APPEND-ONLY EVENT LOG + RUN-STATE REPOSITORY
======================================================
Phase 1 (step 1.2) of RECONSTRUCTION_PLAN.md: the audit/event core.

Layout under the store root (default: <repo_root>/data/runs/ — gitignored,
because real study data must never be committed; see also the data-
protection notes in RECONSTRUCTION_PLAN.md §8.4):

    <root>/<run_id>/events.jsonl   the run's append-only audit log: one
                                    JSON line per event (event classes in
                                    backend/contracts.py), written in
                                    order, never rewritten or deleted
    <root>/<run_id>/state.json     the latest RunState snapshot of the run;
                                    run_version advances by exactly 1 per
                                    save (seed for the optimistic
                                    run_version check in Phase 5.5)
    <root>/study_aborts.jsonl      one tombstone line per study-wide
                                    abort: run_id, aborted_at and cause
                                    ONLY - no participant data (see
                                    RunStateRepository.abort_study)

This module is dependency-pure: stdlib + backend/contracts.py only — no
LLM, no Flask, no global state. All I/O is confined to the explicit
methods below; the legacy pipeline does not use this file.

Used by: the Supervisor and the delivery protocol (Phase 2 onward) for
writes, and by tests / audit review for reads.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from contracts import (
    AgentOutputEvent,
    DeliveryAcknowledgedEvent,
    DeliveryDeliveredEvent,
    DeliveryReservedEvent,
    DeliveryRetransmittedEvent,
    DeliveryValidatedEvent,
    InputReceivedEvent,
    InputTranscribedEvent,
    InterventionSignalEvent,
    RunAbortedEvent,
    RunCompletedEvent,
    RunEvent,
    RunStartedEvent,
    RunState,
    StateTransitionEvent,
    SuicidalitySignal,
    TurnCompletedEvent,
    TurnStartedEvent,
    ValidationDecisionEvent,
    parse_event,
    utc_now,
)

# Default location of all run/event stores on this machine.
# data/runs/ is in .gitignore — audit data stays local and untracked.
DEFAULT_STORE_ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"

# Root-level file holding one tombstone line per study-wide abort (see
# abort_study): the ONLY record kept after the participant's data is deleted.
STUDY_ABORTS_FILENAME = "study_aborts.jsonl"


def _write_json_atomic(path: Path, payload: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a partial state.json."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class EventStore:
    """
    The append-only JSONL event log of one run.

    Append-only is the point (the plan's "binding audit trail"): events
    are only ever appended, in order, with a gapless 1-based event_id
    assigned here. Two things make this durable enough for a research
    study: every append is flushed + fsynced, and readers re-parse every
    line with strict validation (a corrupt line is an error, not a skip).
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The JSONL file this store appends to (and reads from)."""
        return self._path

    def append(self, event: RunEvent) -> RunEvent:
        """
        Append one event as a single JSON line and return the stored
        version (with its assigned event_id).

        If the event already carries an event_id, it must be exactly the
        next expected one — this is how the single-writer invariant
        (no gaps, no duplicates) is enforced mechanically.
        """
        next_seq = self._next_event_id()
        if event.event_id is None:
            stored = event.model_copy(update={"event_id": next_seq})
        elif event.event_id != next_seq:
            raise ValueError(
                f"event_id mismatch: expected {next_seq} (next gapless "
                f"sequence in {self._path}), got {event.event_id}"
            )
        else:
            stored = event

        line = stored.model_dump_json()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return stored

    def read_events(self) -> list[RunEvent]:
        """
        Read the whole log back, in file order, re-validating every line.
        Raises ValueError (with the line number) on a corrupt or
        unrecognized line — an audit log must never silently lose an event.
        """
        events: list[RunEvent] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"corrupt event log: line {line_no} of {self._path} "
                        f"is not valid JSON ({exc})"
                    ) from exc
                try:
                    events.append(parse_event(data))
                except ValueError as exc:
                    raise ValueError(
                        f"corrupt event log: line {line_no} of {self._path} "
                        f"is not a valid run event ({exc})"
                    ) from exc
        return events

    def event_count(self) -> int:
        """Number of events currently in the log."""
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _next_event_id(self) -> int:
        """Next gapless 1-based sequence number for this log."""
        return self.event_count() + 1


class RunStateRepository:
    """
    Where runs live: one directory per run_id under the store root, with
    state.json (latest snapshot) + events.jsonl (the full audit log).

    Version discipline: create_run() requires run_version == 1 and
    save_state() requires exactly current + 1 — a stale or skipped write
    is rejected with a clear message. This is the seed for the
    optimistic run_version check the API gets in Phase 5.5.
    """

    def __init__(self, root_dir: str | Path = DEFAULT_STORE_ROOT):
        self._root = Path(root_dir)

    @property
    def root_dir(self) -> Path:
        return self._root

    def _run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    def state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "state.json"

    def events_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def exists(self, run_id: str) -> bool:
        """True if a run with this id already exists in the repository."""
        return self.state_path(run_id).is_file()

    def create_run(self, state: RunState) -> None:
        """
        Register a new run (reservation, Phase 5.1): writes the initial
        state snapshot and creates the (empty) event log. Fails if the
        run already exists or does not start at run_version 1.
        """
        if state.run_version != 1:
            raise ValueError(
                f"a new run must start at run_version 1, got {state.run_version}"
            )
        if self._run_dir(state.run_id).exists():
            raise FileExistsError(
                f"run '{state.run_id}' already exists at {self._run_dir(state.run_id)}"
            )
        self._run_dir(state.run_id).mkdir(parents=True)
        _write_json_atomic(self.state_path(state.run_id), state.model_dump_json(indent=2))
        self.events_path(state.run_id).touch()

    def load_state(self, run_id: str) -> RunState:
        """Load the latest state snapshot of a run (re-validated on read)."""
        path = self.state_path(run_id)
        if not path.is_file():
            known = ", ".join(self.list_runs()) or "none"
            raise FileNotFoundError(
                f"no run state at {path} (known runs: {known})"
            )
        with path.open("r", encoding="utf-8") as f:
            return RunState.model_validate(json.load(f))

    def save_state(self, run_id: str, state: RunState) -> None:
        """
        Persist a new state snapshot. `state.run_version` must be exactly
        one more than the stored version (single-writer discipline).
        """
        if state.run_id != run_id:
            raise ValueError(
                f"state.run_id {state.run_id!r} does not match requested run {run_id!r}"
            )
        current = self.load_state(run_id)
        if state.run_version != current.run_version + 1:
            raise ValueError(
                f"stale or skipped run_version: stored version is "
                f"{current.run_version}, got {state.run_version} — the "
                f"Supervisor must advance exactly one version per change "
                f"(seed for the optimistic run_version check, Phase 5.5)"
            )
        _write_json_atomic(self.state_path(run_id), state.model_dump_json(indent=2))

    def event_store(self, run_id: str) -> EventStore:
        """The append-only event log of an existing run."""
        if not self.exists(run_id):
            raise FileNotFoundError(
                f"cannot open event store for unknown run '{run_id}' "
                f"(create it first via create_run())"
            )
        return EventStore(self.events_path(run_id))

    def list_runs(self) -> list[str]:
        """All run ids in the repository, sorted."""
        if not self._root.is_dir():
            return []
        return sorted(
            p.name
            for p in self._root.iterdir()
            if p.is_dir() and (p / "state.json").is_file()
        )

    def abort_study(self, run_id: str) -> None:
        """
        Study-wide abort (participant chose to abort the study as a
        whole): delete the run's data - state.json AND the full event
        log (their data) - and keep only their intent to stop the study,
        as one tombstone line in study_aborts.jsonl (run_id, aborted_at,
        cause). The tombstone is written FIRST and fsynced, so a crash
        between the two steps can at worst leave run data behind, never
        the intent record behind nothing.
        """
        if not self.exists(run_id):
            raise FileNotFoundError(
                f"cannot abort unknown run '{run_id}' "
                f"(known runs: {', '.join(self.list_runs()) or 'none'})"
            )
        tombstone = {
            "run_id": run_id,
            "aborted_at": utc_now().isoformat(),
            "cause": "participant_abort_study",
        }
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / STUDY_ABORTS_FILENAME
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(tombstone) + chr(10))  # one JSON line per tombstone
            f.flush()
            os.fsync(f.fileno())
        shutil.rmtree(self._run_dir(run_id))


# ---------------------------------------------------------------------------
# Human-readable rendering of one event (for auditing / review)
# ---------------------------------------------------------------------------


def _short(text: str | None, limit: int = 80) -> str:
    """One-line, length-capped rendering of a (possibly long) text field."""
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) > limit:
        return f'"{flat[:limit]}…"'
    return f'"{flat}"'


def describe_event(event: RunEvent) -> str:
    """
    One human-readable line per event, for reviewing the audit log:
    sequence number, timestamp, event type, turn, and a short detail.
    """
    seq = event.event_id if event.event_id is not None else "?"
    head = f"[{seq}] {event.timestamp.isoformat()} {event.event_type}"
    if event.turn_id:
        head += f" ({event.turn_id})"

    if isinstance(event, RunStartedEvent):
        detail = f"definition={event.definition_version}"
        if event.study_condition is not None:
            detail += f" transparency={event.study_condition.transparency.value}"
    elif isinstance(event, RunCompletedEvent):
        detail = f"definition={event.definition_version} reason={event.reason or '-'}"
    elif isinstance(event, RunAbortedEvent):
        detail = f"cause={event.cause.value} reason={event.reason}"
        if event.feedback is not None:
            detail += f" feedback={_short(event.feedback)}"
    elif isinstance(event, (TurnStartedEvent, TurnCompletedEvent)):
        detail = ""
    elif isinstance(event, InputReceivedEvent):
        detail = f"modality={event.modality.value}"
        if event.content is not None:
            detail += f" content={_short(event.content)}"
        if event.audio_ref is not None:
            detail += f" audio_ref={event.audio_ref}"
    elif isinstance(event, InputTranscribedEvent):
        detail = f"transcript={_short(event.transcript)}"
        if event.language is not None:
            detail += f" language={event.language}"
    elif isinstance(event, StateTransitionEvent):
        subject_id = event.subject_id or event.run_id
        detail = (
            f"{event.subject.value} '{subject_id}': "
            f"{event.from_state} -> {event.to_state} (cause: {event.cause})"
        )
    elif isinstance(event, AgentOutputEvent):
        detail = f"agent={event.agent.value}"
        if event.parse_error is not None:
            detail += f" parse_error={_short(event.parse_error)}"
    elif isinstance(event, ValidationDecisionEvent):
        d = event.decision
        detail = f"outcome={d.outcome.value}"
        if d.reasons:
            detail += f" reasons={[r.code for r in d.reasons]}"
        if d.canonical_fallback_used:
            detail += " canonical_fallback"
    elif isinstance(event, InterventionSignalEvent):
        detail = f"signal={event.signal.signal_type}"
        if isinstance(event.signal, SuicidalitySignal):
            detail += f" action={event.signal.action.value}"
    elif isinstance(event, DeliveryReservedEvent):
        detail = (
            f"delivery={event.delivery_id} kind={event.content_kind.value} "
            f"{_short(event.content_text)}"
        )
    elif isinstance(event, DeliveryValidatedEvent):
        detail = f"delivery={event.delivery_id} outcome={event.outcome.value}"
    elif isinstance(event, DeliveryDeliveredEvent):
        detail = f"delivery={event.delivery_id} modality={event.modality.value}"
    elif isinstance(event, DeliveryAcknowledgedEvent):
        detail = f"delivery={event.delivery_id} ack"
    elif isinstance(event, DeliveryRetransmittedEvent):
        detail = f"delivery={event.delivery_id} attempt {event.attempts}"
    else:  # defensive: a new event type must get its own branch
        detail = f"definition={event.definition_version}"

    return f"{head} :: {detail}" if detail else head
