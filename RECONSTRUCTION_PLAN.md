# Independent Reconstruction Evaluation — Multi-Agent Interview System

> **Basis:**
> - Target: *„Überarbeitete Zielarchitektur für das Multi-Agenten-Interviewsystem"* (PDF, 18 pages, `~/Downloads/`)
> - Reality: repository `local-talking-llm-1` (branch `Z.-Version`, HEAD `dddcd64`)
>
> **Purpose:** This document evaluates — independently of the „Überarbeiteter Migrationsplan" contained in the PDF — what the steps for reconstructing this project should be. The steps in §3 are derived from the target architecture and the actual state of the code. §4 compares the result with the PDF's own migration plan.
>
> Date: 2026-10-19 · Language: English, with German domain terms quoted where relevant

---

## 1. Context

### 1.1 What the target architecture requires (per the PDF)

The corrected target image („Korrigiertes Zielbild") defines six clearly separated
responsibility areas plus audit and delivery infrastructure:

| Component | Role in the target architecture |
|---|---|
| **Supervisor** | Orchestrates the workflow; *sole authority* for binding changes to the session state. Performs no clinical evaluation and produces no participant-facing content |
| **Interventionsmonitor** | Checks every participant input **before** regular interview processing: suicidality, abort intent, prompt injection. Returns three **separate, typed signals** — never merged into one safety score, because the three classes require different reactions |
| **Diagnostiker** | Evaluates an answer only as far as the steering decision requires: answer clarity, extraction of navigation-relevant values, selection of the contextually best next question among all currently required questions. Never formulates a diagnosis — neither internally nor toward the participant |
| **Interview-Navigator** | Deterministic tool and the *authoritative* source for interview structure, required questions, criteria, jumps and state transitions. Decides which questions are admissible; does not diagnose |
| **Sprecher** | Part of the flow **by default**; reformulates content released by Diagnostiker and Navigator to sound more natural without changing diagnostically relevant meaning (content, temporal reference, negation, answer requirement, criterion meaning). Speakerless mode is an explicit, non-default configuration or a technical fallback |
| **Semantischer Validator** | Binding exit control on the Sprecher's version: meaning fidelity, experimental transparency condition, role conformance, non-diagnosis, patient safety. Only a released version may reach the participant |
| **Output layer** | Text / TTS / avatar; delivery is acknowledged (ACK) by the output layer before it is treated as delivered |
| **Ereignis- und Zustandsablage** | Binding audit trail: definition version, agent outputs, validation decisions, state transitions, intervention signals |

Architectural rules any reconstruction must respect:

1. **Four-valued criterion logic** — `SATISFIED`, `NOT_SATISFIED`, `OPEN` (required question not yet asked/answered for the first time) and `UNCLEAR` (an answer exists but could not be interpreted reliably). This distinction separates regular open interview work from targeted clarification needs.
2. **Fail-closed validator** with a documented *canonical bypass*: if the validator blocks, a pre-approved fallback formulation is delivered, so a session can always complete.
3. **Prompt-injection defense in depth**: detection is only an *additional* layer — the architecture must stay safe even when an injection is *not* detected. Separate untrusted user input from system instructions, strictly validate outputs and tool calls, keep agent privileges minimal (OWASP LLM01:2025 / LLM06:2025, AISVS C02, incl. Unicode NFC/NFKC normalization of inputs).
4. **MVP scope** (explicit in the PDF): one section with several question groups incl. `all_required`; a Diagnostiker that evaluates only clarity and navigation-relevant values; a rule engine over `all / any / not / eq / gte / lte / threshold`; the Sprecher as default; the Validator with all check aspects; the Interventionsmonitor with the three signal classes; the canonical bypass; delivery ACK; expert-approved reference scenarios.
5. **Explicitly out of scope** (per the PDF): arbitrary loop structures, diagnostic modules, treatment advice, an LLM-based "free" supervisor, a full authoring frontend.

### 1.2 Current-state snapshot of the repository

| Area | File(s) | Current state |
|---|---|---|
| Orchestration | `backend/app.py` (~535 lines) | Monolithic: CLI args, Flask app, terminal mode, and a single `process_answer()` pipeline. Endpoints: `GET /api/interview/start[/audio]`, `POST /api/chat`, `POST /api/chat/audio` (SSE stream of `audio_chunk`/`done`). No persistence; one participant at a time |
| Interview structure | `backend/interview_guide.py` | **Linear** YAML guide: questions flattened into one list, `_current_index`, at most one LLM follow-up per question. No groups, no criteria, no branching, no state machine |
| LLM usage | `backend/llm_provider.py` | One LangChain chain with in-memory history and a **fixed** `session_id = "voice_assistant_session"`. Model output is a `FEEDBACK: … / FOLLOWUP: …` *text format* parsed with regexes. Second prompt (`get_natural_transition`) only for the `flexible_conversational` profile |
| Safety | `backend/safety_check.py` | One keyword list → one boolean `safety_flag` (non-blocking calm hint on the frontend). Checked on the participant's input only. One category (suicidality); no abort handling, no injection detection, no typed signals |
| Behavior profiles | `backend/behavior_profiles.py`, `study_content/prompts/` | `strict_clinical` (verbatim question delivery) / `flexible_conversational` (natural transition) |
| Modality | `backend/stt.py`, `backend/tts.py`, `backend/speech.py` | faster-whisper STT, Piper TTS, recording/playback glue — cleanly separated from LLM logic |
| Frontend | `frontend/conversation.js` (1145 lines), `index.html` (1056 lines) | Single global `window.CONVERSATION` state; fetch-based; consumes SSE audio; renders the `safety_flag` hint; voice/text conditions, embodiment assets |
| Study content | `study_content/` | Researcher-editable YAML: one example guide (`depression_scid_example.yaml`, **no criteria**), behavior prompts, `safety_keywords.yaml` + `safety_keywords_sources.md`, `study_settings.yaml` (language, active guide, TTS voice) |
| Tests | `tests/` (8 modules, ~700 lines) | `unittest` with `sys.path` hacks; **pytest is not installed**; `Makefile` has no test target; no CI; no LLM test doubles |
| Repo hygiene | — | Working tree **dirty** (`M .gitignore`, `M backend/app.py`, `M requirements.txt`); `diagChatbot/` (the venv) is untracked and **not** gitignored (only `.venv`/`venv/` are); `tts_models/` is properly ignored; `pydantic` 2.11.5 present only transitively via LangChain (not declared in `pyproject.toml`) |

**Reusable as-is (no reconstruction needed):** `stt.py` / `tts.py` / `speech.py`
and the Whisper + Piper stack in general; `study_settings.py` and the
researcher-editable `study_content/` principle; the frontend's SSE audio
consumption logic; the calm, non-blocking safety-hint UX (it becomes the
reaction to the suicidality signal); the local-first data principle (real study
data never goes to a cloud LLM — MiniMax is for testing only).

---

## 2. Gap analysis (target vs. current)

| # | Responsibility area | Today (where) | Target | Gap |
|---|---|---|---|---|
| 1 | Supervisor / single state writer | `process_answer()` in `backend/app.py`; mutable global guide + fixed LLM session id; no events | Single orchestrator, sole writer of session state, every change logged | **Total** — new component |
| 2 | Interventionsmonitor | `safety_check.py`: one keyword list → one boolean `safety_flag`, non-blocking hint, one category | Three separate typed signals (suicidality / abort / prompt injection) checked *before* processing, each with its own reaction and priority rules | **2/3 missing** (abort, injection); the existing channel is not typed, not checked-before-processing, and has no typed reaction |
| 3 | Diagnostiker | `llm_provider.py`: one chain producing feedback + ≤1 follow-up in a regex-parsed text format | Structured `NavigationAssessment` (clarity, navigation values, next-question choice from the candidate set); no diagnosis, ever | **Total** — only "answer feedback" exists today |
| 4 | Interview-Navigator | `interview_guide.py`: linear index over a flattened YAML list | Deterministic, authoritative structure: question groups, criteria, jumps, transitions, `all_required` candidate computation, 4-valued criterion logic | **Total** — new component |
| 5 | Sprecher | Ad hoc: verbatim concatenation (strict profile) or one extra LLM call for transition rewording (flexible profile) | Default component; invariant-preserving reformulation of an `ApprovedContentPackage`; verbatim mode only as explicit config/fallback | **Mostly total** — no content package, no invariants, not the default |
| 6 | Semantischer Validator | — (does not exist) | Binding output gate: meaning fidelity, transparency, role, non-diagnosis, patient safety; ALLOW/REVISE/BLOCK, fail-closed, canonical bypass | **Total** — new component |
| 7 | Delivery protocol | Server sends, client displays; no reservation, no ACK, no idempotency | Reserved → validated → delivered with `delivery_id`, ACK, retransmission, optimistic `run_version` | **Total** — new protocol |
| 8 | Audit / persistence | None (in-memory only; optional `--save-voice` WAV files) | Binding event/state store: definition version, agent outputs, validation decisions, transitions, intervention signals | **Total** — new subsystem |
| 9 | Study condition | Framing text inside the guide YAML; no machine-checkable condition object | `StudyCondition` (transparency etc.) supplied to Speaker *and* Validator; "no speaker" as explicit non-default config | **Mostly total** |
| 10 | Test infrastructure | `unittest` + `sys.path` hacks; no pytest, no CI, no LLM test doubles | Deterministic core tests, state-based navigator tests, contract tests with a fake LLM, expert reference scenarios, fault injection, modality parity | **Total** as infrastructure — must be built *before* phases that rely on it |

**Reading of the gaps:** of the ten areas, eight are total gaps and two are
partially present. Only the modality adapters (STT/TTS/audio), the
researcher-editable content scaffold, and the calm safety-hint UX carry over
unchanged. The reconstruction is therefore primarily a *replacement of the
orchestration and domain layer* (`app.py` + `interview_guide.py` +
`llm_provider.py`), not a rewrite of the I/O layer.

---

## 3. Reconstruction steps — independent evaluation

Ordering principles:

1. **Dependencies first:** contracts → deterministic domain core → LLM agents → monitor → delivery → content.
2. **Deterministic before stochastic:** everything that can be built and tested without a language model (navigator, rule engine, state machine, validator pre-checks) is built *before* any LLM is wired in.
3. **Schema-early, infrastructure-late:** the audit-event schema and the ID scheme (`run_id`/`turn_id`/`delivery_id`) are nearly free in week one and extremely expensive after the Supervisor exists; the heavier persistence concerns (idempotent commands, `run_version`, delivery ACK) come when the delivery protocol does.
4. **Test doubles before contract tests:** a deterministic fake LLM exists before any agent's contract is validated, or contract tests and reference scenarios cannot run reproducibly.

### Phase 0 — Foundation & hygiene

| Step | Action |
|---|---|
| 0.1 | Clean the working tree: commit or consciously resolve the pending changes (`app.py`, `.gitignore`, `requirements.txt`); add `diagChatbot/` to `.gitignore` (or rename the venv to the already-ignored `.venv`) so it cannot be committed by accident |
| 0.2 | Introduce **pytest** (currently absent), a `Makefile test` target, and a green baseline run of the existing `unittest` suite (keep it as-is or migrate it — either way, it must run from one command) |
| 0.3 | Declare `pydantic` explicitly in `pyproject.toml` (today only a transitive dependency); refresh `uv.lock` |
| 0.4 | Write a short baseline spec of the four existing endpoints and their payloads (the "legacy behavior" reference for the later A/B comparison in Phase 8) |

**Acceptance:** `git status` clean; `make test` runs the whole suite green in one
command; `pydantic` is a declared dependency; baseline API behavior documented.

### Phase 1 — Contracts & event core (types before logic)

| Step | Action |
|---|---|
| 1.1 | Pydantic contract models for the complete boundary surface: the 4-valued criterion enum, criterion/question delivery states, `NavigationAssessment`, question-candidate / candidate-set types, `ApprovedContentPackage`, `SpeakerOutput`, `ValidationDecision` (ALLOW / REVISE / BLOCK + reasons), the **three separate** intervention signal types, `StudyCondition` |
| 1.2 | Event-store schema + append-only JSONL writer + run-state repository. One event per: state transition, agent output (raw + parsed), validation decision, intervention signal, delivery lifecycle step. Events carry `definition_version` (from the interview definition) |
| 1.3 | ID scheme from day one: `run_id` (replaces the fixed `session_id`), `turn_id`, `delivery_id`. Per-agent LLM chat history — if kept at all — is keyed by `run_id` |

**Deliverable:** a `contracts` module + a `store` module, both pure, both fully
unit-tested, importable by every later component.

**Acceptance:** every target boundary (see §1.1) is expressible as a validated
type with clear failure messages; a scripted fake run produces a readable,
complete event log.

### Phase 2 — Deterministic navigation core (pure Python, zero LLM)

| Step | Action |
|---|---|
| 2.1 | `InterviewDefinition` loader for the v2 format (question groups, criteria, mandatory meaning units, transitions, `behavior_profile` link) + **import-time validation**: unique IDs, reachable jump targets, no cycles, criteria reference known values/types. Plus a **legacy adapter** mapping today's linear YAML onto the v2 model (so `interview_guide.py` becomes a thin wrapper, not a deletion) |
| 2.2 | `InterviewRun` state machine: question states (UNSEEN / REQUIRED / RESERVED / DELIVERED / ANSWER_RECORDED / CLOSED / SKIPPED) and criterion states (`OPEN / UNCLEAR / SATISFIED / NOT_SATISFIED`), with explicit transition rules and no other mutation path |
| 2.3 | Declarative criteria AST (`all / any / not / eq / gte / lte / threshold`) + evaluation + "decided" computation over the 4-valued logic |
| 2.4 | Navigator service: `all_required` candidate computation, deterministic fallback ordering, selection rules |
| 2.5 | **Supervisor skeleton with stubbed agents** (Diagnostician-stub: "answer recorded, no criterion change"; verbatim Speaker-stub; allow-all Validator-stub; pass-through Monitor-stub): the pipeline *shape* — Monitor → abort-check → Diagnostiker → Navigator → Sprecher → Validator → Output, single writer, all events logged — is fixed before any real LLM is wired in |
| 2.6 | Core test suite: state/transition tests for every legal and illegal transition, criteria truth tables, candidate-set tests, import-validation tests. All deterministic, no network |

**Acceptance:** the core passes its whole test suite with zero LLM calls; a
legacy guide loads through the adapter with identical question order and
behavior; deliberately broken definitions are rejected at import with
actionable messages.

### Phase 3 — LLM agent layer (real models behind the contracts)

| Step | Action |
|---|---|
| 3.1 | `llm_clients.py`: per-agent provider config (provider `ollama`/`minimax`, model, temperature, timeout, retries, max output); **strict structured JSON output with schema validation** against the Phase-1 contracts; and a **deterministic fake LLM** test double |
| 3.2 | Diagnostiker agent: prompt producing a `NavigationAssessment` (answer clarity, navigation-relevant values, next-question choice **restricted to the candidate set** the Navigator handed over), explicitly constrained to navigation-only language — no diagnostic conclusions, ever |
| 3.3 | Sprecher agent: stateless reformulation of an `ApprovedContentPackage` with the invariance requirements (content, temporal reference, negation, answer requirement, criterion meaning); speakerless/"verbatim" mode is an explicit configuration or fallback, **not** the default |
| 3.4 | Validator agent: deterministic pre-checks (regex/heuristics for role claims, diagnostic wording, transparency violations) + a semantic LLM check for meaning fidelity; decision model ALLOW / REVISE (bounded revision loop) / BLOCK; fail-closed on any parse/timeout error; **canonical bypass** (pre-approved fallback formulation) with a logged event |
| 3.5 | Contract tests with the fake LLM (parse success, schema conformance, candidate-set restriction, fail-closed behavior); golden-output integration tests against a real model, marked opt-in (not part of the default run) |

⚠️ **Key early risk, measured here:** a small local model (e.g. `gemma3`) that
fails to produce valid structured output repeatedly would force the
fail-closed/canonical fallback on real runs — a usability and data-quality
problem. Because real study data must never go to a cloud provider, the
validator (and ideally the Diagnostiker) **must run on a local model**; the
structured-output success rate measured in this phase decides the model choice
(and, if necessary, a per-agent model split).

**Acceptance:** the full pipeline runs end-to-end with the fake LLM
deterministically; structured-output success rate measured and documented per
agent against the chosen local model, with the threshold and model decision
recorded.

### Phase 4 — Interventionsmonitor (typed, separate signals)

| Step | Action |
|---|---|
| 4.1 | Reuse `study_content/safety_keywords.yaml` (+ `safety_keywords_sources.md`) as the **SUICIDAL_LANGUAGE** channel: keyword layer first, optionally LLM-assisted refinement; keep the existing calm, non-blocking hint as the participant-facing reaction; log the typed signal |
| 4.2 | **ABORT_REQUEST** channel: detection (keywords + LLM-assisted classification) → a *structured termination protocol* (documented closing message, state `CLOSED`, full audit log persisted) — not a hard crash, not a diagnostic reaction |
| 4.3 | **PROMPT_INJECTION** channel: input preprocessing (Unicode NFC/NFKC normalization, string checks) + LLM-assisted classification; documented explicitly as an *additional layer only* — the architecture's safety guarantees must not depend on it firing |
| 4.4 | Separate, documented reactions per signal class + priority/conflict rules (abort wins over everything; signals are never merged into one score) |

**Acceptance:** each of the three signals can fire independently with its own
reaction and log entry; a scripted abort ends the session gracefully with a
complete audit trail; the pipeline shape from Phase 2.5 is unchanged (the
stub is replaced, not re-plumbed).

### Phase 5 — Delivery protocol & frontend

| Step | Action |
|---|---|
| 5.1 | API v2: reservation on start (returns `run_id`), answer-turn endpoint, delivery carrying `delivery_id`, state endpoint; idempotent answer submission |
| 5.2 | Frontend delivery-ACK in `frontend/conversation.js`: track reserved → delivered (participant actually saw/heard it), retransmit on missing ACK |
| 5.3 | **Modality parity:** text and voice conditions share the same functional event stream and the same interview protocol; the two legacy `/api/chat*` endpoints converge onto it (SSE audio streaming is kept for the voice path) |
| 5.4 | UI: intervention-signal display (reusing the calm non-blocking hint pattern), the abort flow, and the canonical-fallback display when the validator blocks |
| 5.5 | Optimistic `run_version` on the API (client passes the version it last observed; mismatch → re-sync from the state endpoint) |

**Acceptance:** a full session in *both* modalities produces the same sequence
of functional events; killing the client between reserve and ACK does not
corrupt the run (retransmission works); a stale `run_version` is detected and
re-synced.

### Phase 6 — Study condition & content migration

| Step | Action |
|---|---|
| 6.1 | `StudyCondition` object in `study_settings.yaml` (transparency: disclosed AI vs. ambiguous framing; speaker on/off as explicit non-default configuration) — supplied to *both* the Sprecher (as a constraint) and the Validator (as a check aspect) |
| 6.2 | Migrate the example guide (`depression_scid_example.yaml`) to the v2 format with real question groups, criteria and mandatory meaning units — a **prerequisite for every reference scenario**; this requires domain expertise, so it is a parallel workstream, not a coding task |
| 6.3 | Authoring support for the researcher: the import-time validation from 2.1 produces clear, field-level error messages; per the PDF's scope, a full authoring frontend stays out of scope |

**Acceptance:** the v2 example guide validates at import and drives a complete
run through the Supervisor; the StudyCondition is machine-checked by the
Validator (a transparency violation is BLOCKed in tests).

### Phase 7 — Validation workstream (runs in parallel, not a phase)

The PDF's test strategy, scheduled as a workstream that spans Phases 2–8:

- **Expert-approved reference scenarios** — end-to-end scenarios with expected
  state transitions, criterion outcomes and delivered text; the regression
  backbone of every phase. *Requires domain experts → start recruitment early;
  this is a critical path, not a code task.*
- **State-based navigator tests** — systematic event sequences over the
  state/transition model; measure state and transition coverage.
- **Adversarial tests** — prompt injections, prompt leakage attempts, meaning
  drift injected into answers, attempts to make the system diagnose.
- **Fault injection** — LLM timeout, invalid structured output, TTS failure,
  client disconnect mid-delivery.
- **Modality parity tests** — same functional event stream from text and voice.

**Exit criterion for "ready for the study":** all reference scenarios pass in
both modalities; fail-closed behavior of the Validator is demonstrated; a full
run produces a complete, correct audit log.

### Phase 8 — Cutover & operations

| Step | Action |
|---|---|
| 8.1 | Run the legacy pipeline and the new pipeline side by side via a CLI switch (`--pipeline legacy|new`) and compare behavior against the Phase-0.4 baseline spec (A/B comparison, incl. on real audio) |
| 8.2 | Cutover only after the reference scenarios pass on the new pipeline; keep the legacy path available briefly as rollback |
| 8.3 | Operations: structured logging; per-agent LLM health checks; pin the exact local model version(s) and TTS/Whisper model/voice versions in the run record and in `study_settings.yaml`; backups of the run/event store |
| 8.4 | Data protection: reaffirm and enforce the local-only rule for real study data (no MiniMax); document where run stores live and who has read access; document the one-participant-at-a-time operating model as a deliberate constraint |

**Acceptance:** a complete study session runs on the new pipeline with the
legacy pipeline still available for rollback; the run record contains model
and voice versions, and the audit log is complete and reviewable.

---

## 4. Relation to the PDF's „Überarbeiteter Migrationsplan"

The PDF's plan has phases **A** (contracts), **B** (navigator isolation, incl.
legacy loader), **C** (rule engine), **D** (Sprecher + Validator), **E**
(Interventionsmonitor) and **F** (delivery & persistence).

### Adopted from the PDF (my independent assessment converges on these)

- Contracts first (A), navigator isolation with a legacy loader (B),
  declarative criteria AST + 4-valued logic with import validation (C),
  Sprecher as the *default* path with verbatim mode as fallback (D),
  binding validator with fail-closed + canonical bypass (D), three separate
  intervention signals with separate reactions (E), delivery ACK and
  `run_version` (F), and expert reference scenarios as the regression backbone
  instead of happy-path tests.

### Deliberate differences

| Topic | PDF plan | This evaluation | Reason |
|---|---|---|---|
| Persistence / audit store | Phase F (last) | Event schema + append-only store in **Phase 1–2** (only delivery protocol & `run_version` stay late) | The audit schema is the cheapest to design at the start and the most expensive to retrofit after the Supervisor exists; the run store is also the study's data product |
| `run_id` migration | Phase F | ID scheme in **Phase 1** | Cheap early, painful late; the fixed `session_id` should not survive the rebuild |
| Interventionsmonitor | Phase E (after the Validator) | Contract + stub already in **Phase 2** (pipeline shape), detection built in **Phase 4** | The monitor sits *before* everything in the pipeline; fixing the shape early avoids re-plumbing. (Position of the detection work itself is similar to E.) |
| Repo hygiene, test infrastructure (pytest/CI), LLM test doubles | Not part of the plan | **Phase 0** and **Phase 3** | The current repo is dirty, `diagChatbot/` is untracked and un-ignored, pytest is absent, and the plan's promised rule-engine tests and contract tests cannot be executed without them |
| Frontend / API protocol | Mentioned in the codebase mapping (conversation.js sends the Delivery-ACK), but **no frontend step in the plan** | Explicit **Phase 5** | The delivery protocol is worthless without the ACK client side; the API contract change and modality parity need dedicated steps |
| Study content migration (v2 guide with real criteria) & StudyCondition config | Implicit (a „Legacy-Loader" exists, but content authoring and the condition object are never a step) | Explicit **Phase 6**, with the content work flagged as an expert workstream | Without a v2 guide with real criteria, *no* reference scenario can exist — it gates the whole validation workstream |
| Cutover / rollback, operations, data protection, model & voice version pinning | Not part of the plan | **Phase 8** | A research study needs an A/B cutover path, reproducible model/voice versions, and an explicit local-data guarantee |

**In one sentence:** the PDF's plan is a good *feature* plan; this evaluation
adds the *engineering and research-operations* scaffolding around it (Phase 0,
the early event/ID core, the frontend phase, the content/condition migration,
and the cutover), and moves the audit schema to the front.

---

## 5. Risks & open questions

**Risks**

1. **Structured-output reliability of small local models (top risk).**
   The whole agent layer depends on the model returning schema-valid JSON.
   If the chosen local model is unreliable, the fail-closed Validator will
   degrade real sessions into canonical fallbacks. *Mitigation:* measure the
   success rate in Phase 3 before building on top of it; consider a per-agent
   model split (e.g. a larger local model for the Validator); keep the
   deterministic pre-checks strong so the LLM check is a second layer.
2. **Validator must run locally, but local models are the weakest part.**
   Real study data may not leave the machine (MiniMax is test-only), so the
   binding safety gate cannot be outsourced to a stronger cloud model.
   *Mitigation:* early model benchmark (3.5), deterministic pre-checks, and
   accepting that the canonical bypass is the guaranteed-safe floor.
3. **Expert reference scenarios are a critical path.** They need domain
   experts and a v2 guide with real criteria (Phases 6 + 7). If recruitment or
   guide authoring slips, Phases 3–8 have nothing to validate against.
   *Mitigation:* start the content workstream and expert contact in parallel
   with Phase 0/1.
4. **Frontend state complexity.** `conversation.js` holds one large global
   state and already manages audio streaming; adding reservation/ACK/
   retransmission on top risks subtle races (e.g. ACK lost, audio played but
   state not advanced). *Mitigation:* keep the modality adapters and audio
   path untouched; add the delivery protocol as a thin layer with explicit
   unit tests for the race cases.
5. **Dirty working tree + un-ignored venv.** `diagChatbot/` could be committed
   by accident (hundreds of MB), and pending changes to `app.py`/
   `requirements.txt` have no history. *Mitigation:* Phase 0.1 before anything
   else touches the tree.
6. **TTS/Whisper model and Piper voice versions are not pinned anywhere**
   (models are downloaded per machine). For a research study, results
   comparability requires the voice/model versions in the run record.
   *Mitigation:* Phase 8.3.
7. **Flask single-process, one participant at a time.** This is a deliberate
   operating model, not a bug — but the delivery protocol must not silently
   assume concurrency it cannot have. *Mitigation:* keep the constraint
   documented (8.4) and assert it in tests.

**Open questions (need decisions, ideally before Phase 3)**

1. Which local model(s) are the target (gemma3 4B vs. 12B vs. other) and is
   GPU available? Drives risks 1–2.
2. May the Interventionsmonitor use a second LLM call for abort/injection
   classification, or must the MVP stay keyword/regex-only there?
3. Where do run/event stores live (per-participant JSONL files on this
   machine?) and who has read access — data-protection/ethics question, not a
   coding one.
4. Should the Validator deliberately run on a *different* (larger) local
   model than the Diagnostiker/Sprecher?
5. Does the study need the legacy `/api/chat` behavior kept alive at all after
   cutover, or only the interview protocol?

---

## 6. What stays unchanged (deliberately)

- **Modality adapters:** `backend/stt.py`, `backend/tts.py`, `backend/speech.py`
  and the Whisper + Piper stack — clean separation today, no target change.
- **Flask + static-frontend shape**, 127.0.0.1 local server, terminal mode as
  a secondary UI.
- **One participant at a time** — kept and documented as a constraint.
- **`study_content/` as the researcher-only-editable area** (YAML/MD, no code)
  — extended with the v2 guide format and the StudyCondition, not replaced.
- **Calm, non-blocking safety-hint UX** — retained as the participant-facing
  reaction to the suicidality signal (now one of three typed signals).
- **Local-first data principle** — real study data stays on this machine;
  MiniMax remains a test-only provider.
- **`behavior_profiles.py` concept** — the profiles become configuration input
  for the Sprecher (and a constraint for the Validator), not a separate
  prompt-pasting mechanism.

---

*Prepared as an independent evaluation; the source PDF's own migration plan
remains available for side-by-side review (see §4 for the point-by-point
comparison).*
