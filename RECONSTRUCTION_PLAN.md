# Independent Reconstruction Evaluation — Multi-Agent Interview System


> **Basis:**

> - Target: *„Überarbeitete Zielarchitektur für das Multi-Agenten-Interviewsystem"* (PDF, 18 pages, `~/Downloads/`)

> - Reality: repository `local-talking-llm-1` (branch `Z.-Version`, HEAD `dddcd64`)
> - Interview-Navi design note: `docs/interview_navi_design.md` (2026-10-22) — binding for the v2 interview format, the hierarchy and the navigation logic; where the PDF and this note diverge, the note wins

>

> **Purpose:** This document evaluates — independently of the „Überarbeiteter Migrationsplan" contained in the PDF — what the steps for reconstructing this project should be. The steps in §3 are derived from the target architecture and the actual state of the code. §4 compares the result with the PDF's own migration plan.

>

> Date: 2026-10-19 · Language: English, with German domain terms quoted where relevant



---

## 1. Context
### 1.1 What the target architecture requires (per the PDF)

The corrected target image („Korrigiertes Zielbild") defines six clearly separated responsibility areas plus audit and delivery infrastructure:

| Component | Role in the target architecture |
| --------- | ------------------------------- |
| **Supervisor** | Orchestrates the workflow; *sole authority* for binding changes to the session state. Performs no clinical evaluation and produces no participant-facing content |
| **Interventionsmonitor** | Checks every participant input **before** regular interview processing: suicidality, abort intent, prompt injection. Returns three **separate, typed signals** — never merged into one safety score, because the three classes require different reactions |
| **Diagnostiker** | Evaluates an answer only as far as the steering decision requires: answer clarity, extraction of navigation-relevant values, selection of the contextually best next question among all currently required questions. **The only LLM that interacts with the interview as a whole**: judges episode identity (whether two past-episode declarations refer to the same real episode — its verdict arrives as the Navigator's `EpisodeDecision`, applied by the Supervisor through the run's episode-registry operations; the patient profile — Phase 3 — is its context input). Never formulates a diagnosis — neither internally nor toward the participant |
| **Interview-Navigator** („Interview Navi") | Deterministic tool (not an LLM agent) and the *authoritative* source for interview structure — the hierarchical definition (Section Group → Question Group → Single Question), required questions, criteria, jumps and state transitions. Maintains the current „questions to be asked" set; default navigation is the postorder traversal, deviations/cross-links/loops via criteria groups; hands the candidate set to the interviewer agent, which picks the most sensible next question. Does not diagnose |
| **Sprecher** | Part of the flow **by default**; reformulates content released by Diagnostiker and Navigator to sound more natural without changing diagnostically relevant meaning (content, temporal reference, negation, answer requirement, criterion meaning). Speakerless mode is an explicit, non-default configuration or a technical fallback |
| **Semantischer Validator** | Binding exit control on the Sprecher's version: meaning fidelity, experimental transparency condition, role conformance, non-diagnosis, patient safety. Only a released version may reach the participant |
| **Output layer** | Text / TTS / avatar; delivery is acknowledged (ACK) by the output layer before it is treated as delivered |
| **Ereignis- und Zustandsablage** | Binding audit trail: definition version, agent outputs, validation decisions, state transitions, intervention signals |


Architectural rules any reconstruction must respect:

1. **Four-valued criterion logic** — `SATISFIED`, `NOT_SATISFIED`, `OPEN` (required question not yet asked/answered for the first time) and `UNCLEAR` (an answer exists but could not be interpreted reliably). This distinction separates regular open interview work from targeted clarification needs.
2. **Online criterion evaluation** (design note): the interview runs as an online algorithm — criteria are evaluated continuously during the conversation, as soon as the corresponding answers are complete, never after the fact or delayed. Nested criteria are evaluated postorder (question groups before sections); due to the online character, in practice only the next fully-answered criterion is checked at a time.
3. **Fail-closed validator** with a documented *canonical bypass*: if the validator blocks, a pre-approved fallback formulation is delivered, so a session can always complete.
4. **Prompt-injection defense in depth**: detection is only an *additional* layer — the architecture must stay safe even when an injection is *not* detected. Separate untrusted user input from system instructions, strictly validate outputs and tool calls, keep agent privileges minimal
5. **MVP scope**: one section with several question groups — the full v2 format is the Section Group → Question Group → Single Question hierarchy; each question group has a freely configurable internal logic (all-required / sequential / dependent / complex), the default navigation follows the postorder traversal; per-question answer types (yes/no, number, string); a Diagnostiker that evaluates only clarity and navigation-relevant values; a rule engine over the criteria-group operator set (step 2.3: logical NOT/AND/OR, arithmetic equal/unequal/≥/≤/between/min/max, text contains/contains-not, completeness empty/not-empty, group-level aggregation count/threshold/majority/weighted sum — the MVP starts with logical + arithmetic + count/threshold); the Sprecher as default; the Validator with all check aspects; the Interventionsmonitor with the three signal classes; the canonical bypass; delivery ACK; expert-approved reference scenarios.
6. **Explicitly out of scope**: arbitrary *uncontrolled* loop structures — loops *are* part of the design via criteria groups (deviations, cross-links, loops); the concrete behavior of a loop leading back to an already-answered (locked) group was the open point (§5, question 6) and was **decided 2026-10-26 as the EPISODE MODEL** (per-interview `episodes` block; episode-aspect jumps re-ask the locked target group for another episode; plain jumps to a locked target are ignored + logged; jumps are deferred to group completion — see steps 2.1/2.2/2.4 and the design-note addendum), diagnostic modules, treatment advice, an LLM-based "free" supervisor. A full authoring *frontend* is not part of the MVP but a foreseen later input path (researcher → frontend → backend converts the interview into a saved navigable object; YAML files remain the input path until then).


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


**Reusable as-is (no reconstruction needed):** `stt.py` / `tts.py` / `speech.py` and the Whisper + Piper stack in general; `study_settings.py` and the researcher-editable `study_content/` principle; the frontend's SSE audio consumption logic; the calm, non-blocking safety-hint UX (it becomes the reaction to the suicidality signal); the local-first data principle (real study data never goes to a cloud LLM — MiniMax is for testing only).

---

## 2. Gap analysis (target vs. current)

| # | Responsibility area | Today (where) | Target | Gap |
|---|---|---|---|---|
| 1 | Supervisor / single state writer | `process_answer()` in `backend/app.py`; mutable global guide + fixed LLM session id; no events | Single orchestrator, sole writer of session state, every change logged | **Total** — new component |
| 2 | Interventionsmonitor | `safety_check.py`: one keyword list → one boolean `safety_flag`, non-blocking hint, one category | Three separate typed signals (suicidality / abort / prompt injection) checked *before* processing, each with its own reaction and priority rules | **2/3 missing** (abort, injection); the existing channel is not typed, not checked-before-processing, and has no typed reaction |
| 3 | Diagnostiker | `llm_provider.py`: one chain producing feedback + ≤1 follow-up in a regex-parsed text format | Structured `NavigationAssessment` (clarity, navigation values, next-question choice from the candidate set); no diagnosis, ever | **Total** — only "answer feedback" exists today |
| 4 | Interview-Navigator | `interview_guide.py`: linear index over a flattened YAML list | Deterministic, authoritative tool („Interview Navi"): hierarchical structure (Section Group / Question Group / Single Question), criteria groups, jumps, postorder default navigation, candidate-set computation, 4-valued criterion logic with online evaluation | **Total** — new component |
| 5 | Sprecher | Ad hoc: verbatim concatenation (strict profile) or one extra LLM call for transition rewording (flexible profile) | Default component; invariant-preserving reformulation of an `ApprovedContentPackage`; verbatim mode only as explicit config/fallback | **Mostly total** — no content package, no invariants, not the default |
| 6 | Semantischer Validator | — (does not exist) | Binding output gate: meaning fidelity, transparency, role, non-diagnosis, patient safety; ALLOW/REVISE/BLOCK, fail-closed, canonical bypass | **Total** — new component |
| 7 | Delivery protocol | Server sends, client displays; no reservation, no ACK, no idempotency | Reserved → validated → delivered with `delivery_id`, ACK, retransmission, optimistic `run_version` | **Total** — new protocol |
| 8 | Audit / persistence | None (in-memory only; optional `--save-voice` WAV files) | Binding event/state store: definition version, agent outputs, validation decisions, transitions, intervention signals | **Total** — new subsystem |
| 9 | Study condition | Framing text inside the guide YAML; no machine-checkable condition object | `StudyCondition` (transparency etc.) supplied to Speaker *and* Validator; "no speaker" as explicit non-default config | **Mostly total** |
| 10 | Test infrastructure | `unittest` + `sys.path` hacks; no pytest, no CI, no LLM test doubles | Deterministic core tests, state-based navigator tests, contract tests with a fake LLM, expert reference scenarios, fault injection, modality parity | **Total** as infrastructure — must be built *before* phases that rely on it |

**Reading of the gaps:** of the ten areas, eight are total gaps and two are partially present. Only the modality adapters (STT/TTS/audio), the researcher-editable content scaffold, and the calm safety-hint UX carry over unchanged. The reconstruction is therefore primarily a *replacement of the orchestration and domain layer* (`app.py` + `interview_guide.py` + `llm_provider.py`), not a rewrite of the I/O layer.

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

**Acceptance:** `git status` clean; `make test` runs the whole suite green in one command; `pydantic` is a declared dependency; baseline API behavior documented.

### Phase 1 — Contracts & event core (types before logic)

| Step | Action |
|---|---|
| 1.1 | Pydantic contract models for the complete boundary surface: the 4-valued criterion enum, criterion/question delivery states, `NavigationAssessment`, question-candidate / candidate-set types, `ApprovedContentPackage`, `SpeakerOutput`, `ValidationDecision` (ALLOW / REVISE / BLOCK + reasons), the **three separate** intervention signal types, `StudyCondition` |
| 1.2 | Event-store schema + append-only JSONL writer + run-state repository. One event per: state transition, agent output (raw + parsed), validation decision, intervention signal, delivery lifecycle step. Events carry `definition_version` (from the interview definition) |
| 1.3 | ID scheme from day one: `run_id` (replaces the fixed `session_id`), `turn_id`, `delivery_id`. Per-agent LLM chat history — if kept at all — is keyed by `run_id` |

**Deliverable:** a `contracts` module + a `store` module, both pure, both fully unit-tested, importable by every later component.

**Acceptance:** every target boundary (see §1.1) is expressible as a validated type with clear failure messages; a scripted fake run produces a readable, complete event log.

### Phase 2 — Deterministic navigation core (pure Python, zero LLM)

| Step | Action |
|---|---|
| 2.1 | `InterviewDefinition` loader for the v2 format — the **hierarchical structure** (design note): Section Groups (arbitrarily nestable; configurable relations to other sections: temporal / pathological / free), Question Groups (freely configurable internal logic: all-required / sequential / dependent / complex; markable as a criteria group), Single Questions (ID, question text, answer type yes/no / number / string, mandatory meaning units); declared navigation values, criteria (a criteria group's criterion always refers to *all* elements of the group together; section groups can be criteria groups too), transitions (jumps as direct ID references to any question / question group / section group; a jump may also declare `force_criterion` — a declarative coding decision the Supervisor applies via `run.force_criterion` when the jump fires, design-note addendum 2026-11-03, e.g. the SCID A52 "code A30 with '+' / A38 with 'JA'"), `behavior_profile` link. **ID scheme**: `sec_*` / `qg_*` / `sq_*` — the prefix is mandatory (validates the element's type); the numeric position part encodes the position in the hierarchy but is INFORMATIONAL (assigned at creation, never renumbered or re-validated on edits, so ids stay stable). **Import-time validation**: unique, type-consistent (prefix) IDs; jump targets exist (dead ends — missing or invalid jump targets — detected and prevented *at import*); criteria reference known values/types; at least one question group with at least one question. Loops are allowed (the earlier loop-free requirement is dropped); the loop-back-to-locked-group behavior was the open point (§5 question 6) and was **decided 2026-10-26 as the EPISODE MODEL**: the v2 format gained the per-interview `episodes` block (now-episode with window/rule; past-episode declaration `dynamic` — declared at run time — or `predeclared` with fixed names), the `episode_evaluable` marker on question groups, and the jump's episode aspect (`episode_jump: true` / `episode: <name>` — exactly one; import-time validation ties all three together: disabled-mode rejection, mode fit, predeclared names non-empty/unique/non-blank, target must be an `episode_evaluable` question group). Plus a **legacy adapter** mapping today's linear YAML onto the v2 model (so `interview_guide.py` becomes a thin wrapper, not a deletion) |
| 2.2 | `InterviewRun` state machine: question states (UNSEEN / REQUIRED / RESERVED / DELIVERED / ANSWER_RECORDED / CLOSED / SKIPPED) and criterion states (`OPEN / UNCLEAR / SATISFIED / NOT_SATISFIED`), with explicit transition rules and no other mutation path. A question is asked at most once PER EPISODE (the loop question was **decided 2026-10-26 as the EPISODE MODEL**, design-note addendum): the run seeds the per-interview episode registry (always the `now` episode; predeclared past episodes by name; dynamic ones at run time with deterministic ids `ep_1`, `ep_2`, …), every question's first visit is the now-episode, and the (question, canonical episode) pair is recorded in the run (`question_episodes`). The plain "never re-asked" rule is the special case of it: `CLOSED/SKIPPED → REQUIRED` is the only transition out of a terminal question state and exists solely for the episode re-ask (`reopen_question`, rejected if the question was already asked for that real episode). The run-level episode operations `declare_episode` / `record_episode_alias` (the patient's verbatim framings) / `merge_episodes` (the Diagnostiker's identity verdict; the earlier declaration is canonical, `merged_into` records the merge) are logged state transitions. REVISED is the clarification cycle of one still-open question, not a navigation re-ask. **(WInf-inspired additions, step 2.5 steering):** the run-level status includes `PAUSED` (the participant's technical-failure pause — `ACTIVE → PAUSED` / `PAUSED → ACTIVE`; a paused run is frozen but resumable, `COMPLETED`/`ABORTED` stay the only terminal states), and the question table includes `DELIVERED → SKIPPED` (the participant-initiated skip — explicit skip command / refusal intent, **only once the skip-probe threshold is reached**: the question is probed first — ask why + alternatives — a skip is never a negative answer) and `DELIVERED → REVISED` (the **skip-probe re-ask**: the participant asked to skip / refused BEFORE answering anything, so the question is re-offered with the probe — ask why + alternatives — while no answer is recorded yet, `REVISED → DELIVERED` again) |
| 2.3 | Declarative criteria AST + evaluation + "decided" computation over the 4-valued logic. The linking logic per criteria group supports freely combinable operators in five categories — **logical** (NOT, AND, OR), **arithmetic on a single value** (equal, unequal, ≥, ≤, between, min, max), **text** (contains, contains-not), **completeness** (empty, not-empty), **group-level aggregation** (count, threshold, majority, weighted sum); a criteria group's criterion always refers to *all* elements of the group together. Evaluation is **online** (a criterion is evaluated as soon as its answers are complete) and runs **postorder** over the hierarchy (question groups before sections) |
| 2.4 | Navigator service (the „Interview Navi" tool the interviewer agent uses): maintains the current „questions to be asked" set; **default navigation = postorder traversal** over the hierarchy (without criteria groups); with criteria groups: deviations, cross-links and loops; after a jump, the postorder traversal is followed again by default; the per-group internal logic (all-required / sequential / dependent / complex) drives the candidate computation; deterministic fallback ordering, selection rules — the interviewer agent picks the most sensible next question from the candidate set and reacts to the criteria-group outcomes. **Jumps are DEFERRED** (owner decision 2026-10-26): criteria stay decided online, but jumps and group consequences apply only at GROUP COMPLETION — `complete_group` / `group_is_complete` (skips for the group's remaining open optional questions + the first declared jump that fires; `resolve_jump` mid-group fails loudly). The locked-target loop-back is the EPISODE MODEL: a plain jump to a locked group is ignored + logged (`loop_note`); an episode jump re-asks the locked target group for that episode (`reopen_proposals` + `reframe_context`); a dynamic episode jump first reports `pending_episode` and awaits the Diagnostiker's `EpisodeDecision` (re-called on `resolve_jump`) |
| 2.5 | **Supervisor skeleton with stubbed agents** (Diagnostician-stub: "answer recorded, no criterion change"; verbatim Speaker-stub; allow-all Validator-stub; pass-through Monitor-stub): the pipeline *shape* — Monitor (signals + **input intent**) → failure-choice check → abort-check → the **deterministic skip-REQUEST pre-checks** (explicit skip command / refusal intent: the question is PROBED first — ask why + alternatives — and skipped only once the skip-probe threshold is reached) → Diagnostiker (the **technical-failure policy** on a model error) → the **deterministic steering decision** → Navigator → Sprecher → Validator → Output, single writer, all events logged — is fixed before any real LLM is wired in. **(WInf-inspired steering & failure policy, implemented with step 2.5):** the Interventionsmonitor measures the input's *intent* (`InputIntent`: answer / clarification_request / cannot_answer / refusal / off_topic / unclear — the intent lives at the monitor, NOT the Diagnostiker); the DETERMINISTIC reaction table (`interview_navigator.steering_decision` — intent + clarity + **per-kind re-ask counters** + the config-driven thresholds of `SteeringConfig`, researcher-editable via `study_content/steering_settings.yaml`) decides PROCEED / REASK (clarify / redirect / gentle / rephrase — each kind with its own counter and threshold; the rephrase re-ask is unlimited by default, WInf RT01, and does not consume the clarification budget — plus the **skip-probe** kind: the skip REQUEST (explicit skip command / refusal intent, `skip_request_decision`, shared by the command pre-check and the refusal branch of the table) is probed while `max_skip_probes` (default 1; 0 = immediate skip) allows, the question is re-asked (`DELIVERED → REVISED → DELIVERED`, FOLLOWUP with the pre-approved `skip_probe_text`: ask why + the alternative — a vague/short answer is fine) and NO readings are recorded) / SKIP (the skip request once the probe threshold is reached — explicit / refused — or the budget exhausted — unresolved); a **skipped question is NEVER a negative answer** (no readings, the criteria depending on it stay OPEN/UNCLEAR; the participant-initiated skip is the `DELIVERED → SKIPPED` transition); a **technical (model) failure is never a participant action** (no state change, no budget consumed, `model_failure` audit event, the participant is asked to resend; after the configured number of consecutive failures — default 3 — the PARTICIPANT chooses: continue (the counter resets) or **pause the run** (`RunStatus.PAUSED` — frozen, resumable via `resume_run()`, the pending question is re-asked)). The LLM-free stubs degrade the table to the classic clarity-only path (the stub intent is always `answer`) |
| 2.6 | Core test suite: state/transition tests for every legal and illegal transition, criteria truth tables, candidate-set tests, import-validation tests. All deterministic, no network |

**Acceptance:** the core passes its whole test suite with zero LLM calls; a legacy guide loads through the adapter with identical question order and behavior; deliberately broken definitions are rejected at import with actionable messages.

### Phase 3 — LLM agent layer (real models behind the contracts)

| Step | Action |
|---|---|
| 3.1 | `llm_clients.py`: per-agent provider config (provider `ollama`/`minimax`, model, temperature, timeout, retries, max output); **strict structured JSON output with schema validation** against the Phase-1 contracts; and a **deterministic fake LLM** test double |
| 3.2 | Diagnostiker agent: prompt producing a `NavigationAssessment` (answer clarity, navigation-relevant values, next-question choice **restricted to the candidate set** the Navigator handed over), explicitly constrained to navigation-only language — no diagnostic conclusions, ever. **(WInf-inspired steering context, wired with the step 2.5 steering table):** the prompt carries the re-ask context of a REVISED question (the re-ask *kind* — clarify / redirect / gentle / rephrase — and how many attempts of that kind were already given), and a re-asked answer is evaluated **together with the participant's earlier answer(s) for that question** (the item-turn history, WInf `item_turns` — the clarification completes the first answer, it does not replace it). The input's *intent* is NOT part of this prompt — it is measured by the Interventionsmonitor (Phase 4.5) and consumed by the deterministic steering table, so the Diagnostiker stays a pure answer evaluator |
| 3.3 | Sprecher agent: stateless reformulation of an `ApprovedContentPackage` with the invariance requirements (content, temporal reference, negation, answer requirement, criterion meaning); speakerless/"verbatim" mode is an explicit configuration or fallback, **not** the default. **Not-yet-implementable (WInf-inspired) — future acknowledgement behavior, documented here as the target spec:** the Sprecher's participant-facing reactions (the re-asks of the steering table, the failure/resend prompts once they are generated rather than pre-approved) must be **non-generic acknowledgements** — the reaction connects to the participant's ACTUAL preceding utterance (a redirect names what was off-topic, the gentle follow-up names the question the participant could not answer, the rephrase re-frames THAT question, the skip probe references the participant's ACTUAL skip request / refusal and the alternatives they could take), never a canned "okay, next question". Alongside it: a **rejected-phrases list** (researcher-maintained: formulations that must never appear — generic dismissals, symptom labels, diagnoses, clinical jargon) that the prompt is built against, and an **empathy fallback**: when the monitor's signals/intent mark the input as distressed (suicidality HINT_ONLY, refusal, repeated cannot_answer), the Sprecher uses the calm, empathic fallback formulation instead of a procedural one (the calm, non-blocking safety-hint UX of §6 is retained as its participant-facing form). *Why not now:* this needs the real LLM Sprecher (this step), the monitor's intent/signal channels (Phase 4), and a researcher-approved phrase inventory — the deterministic step 2.5 steering ships with the PRE-APPROVED fixed texts (`SteeringConfig`), which are the fail-closed default until all three exist |
| 3.4 | Validator agent: deterministic pre-checks (regex/heuristics for role claims, diagnostic wording, transparency violations) + a semantic LLM check for meaning fidelity; decision model ALLOW / REVISE (bounded revision loop) / BLOCK; fail-closed on any parse/timeout error; **canonical bypass** (pre-approved fallback formulation) with a logged event. **Not-yet-implementable (WInf-inspired) — future acknowledgement-quality checks, documented here as the target spec:** the Validator gains an **acknowledgement aspect** (a new `ValidationAspect` value, added with this step) that checks the SpeakerOutput against the 3.3 target spec: (a) **non-generic** — the acknowledgement must reference the participant's preceding input (a verbatim echo of the generic "okay/understood/next question" across consecutive turns is a REVISE), (b) **rejected phrases** — the researcher-maintained list is enforced deterministically (a hit is a BLOCK with a stable reason code), (c) **empathy fallback** — on a distressed input (the monitor's signal/intent, passed with the `ApprovedContentPackage`) the output must match the calm fallback register (a procedural reaction to distress is a REVISE). *Why not now:* it checks what the 3.3 real Sprecher produces — with the step 2.5 stubs the check has nothing to bite on (the pre-approved fixed texts bypass the Speaker/Validator by design, like the terminal texts) |
| 3.5 | Contract tests with the fake LLM (parse success, schema conformance, candidate-set restriction, fail-closed behavior); golden-output integration tests against a real model, marked opt-in (not part of the default run) |


⚠️ **Key early risk, measured here:** a small local model (e.g. `gemma3`) that fails to produce valid structured output repeatedly would force the fail-closed/canonical fallback on real runs — a usability and data-quality problem. Because real study data must never go to a cloud provider, the validator (and ideally the Diagnostiker) **must run on a local model**; the structured-output success rate measured in this phase decides the model choice (and, if necessary, a per-agent model split).

**Acceptance:** the full pipeline runs end-to-end with the fake LLM deterministically; structured-output success rate measured and documented per agent against the chosen local model, with the threshold and model decision recorded.

### Phase 4 — Interventionsmonitor (typed, separate signals)

| Step | Action |
|---|---|
| 4.1 | Reuse `study_content/safety_keywords.yaml` (+ `safety_keywords_sources.md`) as the **SUICIDAL_LANGUAGE** channel: keyword layer first, optionally LLM-assisted refinement; keep the existing calm, non-blocking hint as the participant-facing reaction; log the typed signal |
| 4.2 | **ABORT_REQUEST** channel: detection (keywords + LLM-assisted classification) → a *structured termination protocol* (documented closing message, state `CLOSED`, full audit log persisted) — not a hard crash, not a diagnostic reaction |
| 4.3 | **PROMPT_INJECTION** channel: input preprocessing (Unicode NFC/NFKC normalization, string checks) + LLM-assisted classification; documented explicitly as an *additional layer only* — the architecture's safety guarantees must not depend on it firing |
| 4.4 | Separate, documented reactions per signal class + priority/conflict rules (abort wins over everything; signals are never merged into one score) |
| 4.5 | **Input-intent measurement (WInf-inspired):** the Interventionsmonitor's second job — it classifies every input's *intent* (contracts.`InputIntent`: answer / clarification_request / cannot_answer / refusal / off_topic / unclear; the `MonitorResult` return type already carries it) via keywords/heuristics (and, if open question 2 allows, LLM-assisted classification). The intent is consumed ONLY by the deterministic steering reaction table (step 2.5) and the two LLM-free skip-REQUEST pre-checks (explicit skip command / refusal — both probed first: ask why + alternatives, skip only at the probe threshold) — the Diagnostiker never sees it, and the intent never changes a signal's reaction (the three signal classes keep their strict separation) |


**Acceptance:** each of the three signals can fire independently with its own reaction and log entry; a scripted abort ends the session gracefully with a complete audit trail; the pipeline shape from Phase 2.5 is unchanged (the stub is replaced, not re-plumbed); the intent channel (4.5) drives the deterministic steering table in the scripted tests exactly as the step 2.5 fakes do (refusal → skip-probe re-ask then skip at the threshold, off-topic → one redirect then skip, cannot-answer → one gentle follow-up then skip, clarification-request → rephrase re-ask), with the real keyword/LLM classification instead of the scripted intents.

### Phase 5 — Delivery protocol & frontend

| Step | Action |
|---|---|
| 5.1 | API v2: reservation on start (returns `run_id`), answer-turn endpoint, delivery carrying `delivery_id`, state endpoint; idempotent answer submission |
| 5.2 | Frontend delivery-ACK in `frontend/conversation.js`: track reserved → delivered (participant actually saw/heard it), retransmit on missing ACK |
| 5.3 | **Modality parity:** text and voice conditions share the same functional event stream and the same interview protocol; the two legacy `/api/chat*` endpoints converge onto it (SSE audio streaming is kept for the voice path) |
| 5.4 | UI: intervention-signal display (reusing the calm non-blocking hint pattern), the abort flow, and the canonical-fallback display when the validator blocks |
| 5.5 | Optimistic `run_version` on the API (client passes the version it last observed; mismatch → re-sync from the state endpoint) |



**Acceptance:** a full session in *both* modalities produces the same sequence of functional events; killing the client between reserve and ACK does not corrupt the run (retransmission works); a stale `run_version` is detected and re-synced.


### Phase 6 — Study condition & content migration

| Step | Action |
|---|---|
| 6.1 | `StudyCondition` object in `study_settings.yaml` (transparency: disclosed AI vs. ambiguous framing; speaker on/off as explicit non-default configuration) — supplied to *both* the Sprecher (as a constraint) and the Validator (as a check aspect) |
| 6.2 | Migrate the example guide (`depression_scid_example.yaml`) to the v2 format (Section / Question Group / Single Question hierarchy, criteria groups with group-level operators, per-question answer types) with real criteria and mandatory meaning units — a **prerequisite for every reference scenario**; this requires domain expertise, so it is a parallel workstream, not a coding task |
| 6.3 | Authoring support for the researcher: the import-time validation from 2.1 produces clear, field-level error messages; a full authoring frontend is not part of the MVP but a foreseen later input path (researcher → frontend → backend converts the interview into a saved navigable object) — YAML files remain the input path until then |

**Acceptance:** the v2 example guide validates at import and drives a complete run through the Supervisor; the StudyCondition is machine-checked by the Validator (a transparency violation is BLOCKed in tests).


### Phase 7 — Validation workstream (runs in parallel, not a phase)

The PDF's test strategy, scheduled as a workstream that spans Phases 2–8:

- **Expert-approved reference scenarios** — end-to-end scenarios with expected state transitions, criterion outcomes and delivered text; the regression backbone of every phase. *Requires domain experts → start recruitment early; this is a critical path, not a code task.*

- **State-based navigator tests** — systematic event sequences over the state/transition model; measure state and transition coverage.

- **Adversarial tests** — prompt injections, prompt leakage attempts, meaning drift injected into answers, attempts to make the system diagnose.

- **Fault injection** — LLM timeout, invalid structured output, TTS failure, client disconnect mid-delivery. *(The LLM-failure part is already wired to the deterministic technical-failure policy of step 2.5 — WInf-inspired: no state change, no budget consumed, the `model_failure` audit event, the resend prompt, and the participant's continue/pause choice after the threshold with the `PAUSED` run state; exercised deterministically with the scripted failing diagnostiker in `tests/test_steering.py`. The TTS-failure and client-disconnect cases arrive with the delivery protocol (Phase 5.2).)*

- **Modality parity tests** — same functional event stream from text and voice.

**Exit criterion for "ready for the study":** all reference scenarios pass in both modalities; fail-closed behavior of the Validator is demonstrated; a full run produces a complete, correct audit log.

### Phase 8 — Cutover & operations

| Step | Action |
|---|---|
| 8.1 | Run the legacy pipeline and the new pipeline side by side via a CLI switch (`--pipeline legacy|new`) and compare behavior against the Phase-0.4 baseline spec (A/B comparison, incl. on real audio) |
| 8.2 | Cutover only after the reference scenarios pass on the new pipeline; keep the legacy path available briefly as rollback |
| 8.3 | Operations: structured logging; per-agent LLM health checks; pin the exact local model version(s) and TTS/Whisper model/voice versions in the run record and in `study_settings.yaml`; backups of the run/event store |
| 8.4 | Data protection: reaffirm and enforce the local-only rule for real study data (no MiniMax); document where run stores live and who has read access; document the one-participant-at-a-time operating model as a deliberate constraint |

**Acceptance:** a complete study session runs on the new pipeline with the legacy pipeline still available for rollback; the run record contains model and voice versions, and the audit log is complete and reviewable.

---

## 4. Relation to the PDF's „Überarbeiteter Migrationsplan"
The PDF's plan has phases **A** (contracts), **B** (navigator isolation, incl. legacy loader), **C** (rule engine), **D** (Sprecher + Validator), **E** (Interventionsmonitor) and **F** (delivery & persistence).

### Adopted from the PDF (my independent assessment converges on these)

- Contracts first (A), navigator isolation with a legacy loader (B), declarative criteria AST + 4-valued logic with import validation (C), Sprecher as the *default* path with verbatim mode as fallback (D), binding validator with fail-closed + canonical bypass (D), three separate intervention signals with separate reactions (E), delivery ACK and `run_version` (F), and expert reference scenarios as the regression backbone instead of happy-path tests.

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

**In one sentence:** the PDF's plan is a good *feature* plan; this evaluation adds the *engineering and research-operations* scaffolding around it (Phase 0, the early event/ID core, the frontend phase, the content/condition migration, and the cutover), and moves the audit schema to the front.

---

## 5. Risks & open questions
**Risks**

1. **Structured-output reliability of small local models (top risk).** The whole agent layer depends on the model returning schema-valid JSON. If the chosen local model is unreliable, the fail-closed Validator will degrade real sessions into canonical fallbacks. *Mitigation:* measure the success rate in Phase 3 before building on top of it; consider a per-agent model split (e.g. a larger local model for the Validator); keep the deterministic pre-checks strong so the LLM check is a second layer.

2. **Validator must run locally, but local models are the weakest part.** Real study data may not leave the machine (MiniMax is test-only), so the binding safety gate cannot be outsourced to a stronger cloud model.  *Mitigation:* early model benchmark (3.5), deterministic pre-checks, and accepting that the canonical bypass is the guaranteed-safe floor.

3. **Expert reference scenarios are a critical path.** They need domain experts and a v2 guide with real criteria (Phases 6 + 7). If recruitment or guide authoring slips, Phases 3–8 have nothing to validate against. *Mitigation:* start the content workstream and expert contact in parallel with Phase 0/1.

4. **Frontend state complexity.** `conversation.js` holds one large global state and already manages audio streaming; adding reservation/ACK/ retransmission on top risks subtle races (e.g. ACK lost, audio played but state not advanced). *Mitigation:* keep the modality adapters and audio path untouched; add the delivery protocol as a thin layer with explicit unit tests for the race cases.

5. **Dirty working tree + un-ignored venv.** `diagChatbot/` could be committed by accident (hundreds of MB), and pending changes to `app.py`/ `requirements.txt` have no history. *Mitigation:* Phase 0.1 before anything else touches the tree.

6. **TTS/Whisper model and Piper voice versions are not pinned anywhere** (models are downloaded per machine). For a research study, results comparability requires the voice/model versions in the run record. *Mitigation:* Phase 8.3.

7. **Flask single-process, one participant at a time.** This is a deliberate operating model, not a bug — but the delivery protocol must not silently assume concurrency it cannot have. *Mitigation:* keep the constraint documented (8.4) and assert it in tests.


**Open questions (need decisions, ideally before Phase 3)**

1. Which local model(s) are the target (gemma3 4B vs. 12B vs. other) and is GPU available? Drives risks 1–2.

2. May the Interventionsmonitor use a second LLM call for abort/injection classification, or must the MVP stay keyword/regex-only there?

3. Where do run/event stores live (per-participant JSONL files on this machine?) and who has read access — data-protection/ethics question, not a coding one.

4. Should the Validator deliberately run on a *different* (larger) local model than the Diagnostiker/Sprecher?

5. Does the study need the legacy `/api/chat` behavior kept alive at all after cutover, or only the interview protocol?

6. ~~Behavior of loops leading back to an already-answered (locked) group~~ — **DECIDED (2026-10-26)**: the EPISODE MODEL (design-note addendum `docs/interview_navi_design.md`, implemented in steps 2.1/2.2/2.4). Per-interview `episodes` block (now-episode + past-episode declaration: dynamic / predeclared); a jump with an episode aspect re-asks the locked target group for that episode, a plain jump is ignored + logged; the same question is never re-asked for the same real episode (the (question, episode) once-only invariant, `CLOSED/SKIPPED → REQUIRED` as the episode re-open). Episode identity is judged by the Diagnostiker — the only LLM that interacts with the interview as a whole (its NAVIGATION prompt mode, Phase 3; verdict as `EpisodeDecision`, applied by the Supervisor via the run's episode-registry operations). Jumps are deferred to group completion. The patient-profile direction stays Phase 3 (context input to the Diagnostiker's identity judgment — it does not block this decision).

7. **Persistence of the steering / failure-policy counters.** The step 2.5 steering bookkeeping (the per-kind re-ask counters of the in-flight question, the consecutive technical-failure count, the pending continue/pause choice) is currently SUPERVISOR-OPERATIONAL state (in memory, like the turn/delivery bookkeeping): it survives the run's life but not a process restart, and it is NOT part of the persisted `RunState` snapshot. The AUDIT TRAIL of both policies is complete regardless (the `model_failure` events with agent + detail, the skip/revise transition causes carrying the WInf skip reason and the kind's attempt number, the `ACTIVE ↔ PAUSED` transitions, the pre-approved-text deliveries). *Decision needed:* do the counters (and the pending choice) deserve a place in `RunState` — for crash-consistent resumability of a mid-run pause across a server restart (Phase 5.5 re-sync), or is the audit trail + the state machine (the pending question is still `DELIVERED`, the run is `PAUSED`) sufficient to re-derive everything on reload? Leaning: keep it operational until Phase 5 (the delivery protocol needs a restart-recovery decision anyway), and only then promote what the re-sync endpoint has to answer.

---

## 6. What stays unchanged (deliberately)

- **Modality adapters:** `backend/stt.py`, `backend/tts.py`, `backend/speech.py` and the Whisper + Piper stack — clean separation today, no target change.

- **Flask + static-frontend shape**, 127.0.0.1 local server, terminal mode as a secondary UI.

- **One participant at a time** — kept and documented as a constraint.

- **`study_content/` as the researcher-only-editable area** (YAML/MD, no code) — extended with the v2 guide format, the StudyCondition and the steering policy (`steering_settings.yaml`, step 2.5), not replaced.

- **Calm, non-blocking safety-hint UX** — retained as the participant-facing reaction to the suicidality signal (now one of three typed signals).

- **Local-first data principle** — real study data stays on this machine; MiniMax remains a test-only provider.

- **`behavior_profiles.py` concept** — the profiles become configuration input for the Sprecher (and a constraint for the Validator), not a separate prompt-pasting mechanism.
