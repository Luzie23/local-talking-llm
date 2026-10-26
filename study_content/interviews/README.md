# interviews/

This folder holds the interview guides — one file per guide (for example
a depression screening guide, or a general mental-health guide). Each file
lists the fixed questions in order, exactly as the assistant presents them
to the participant, plus the framing message shown at the start and the
closing message shown at the end.

**Two formats are supported** (auto-detected by
`backend/interview_definition.py`: a top-level `definition_version` key
means v2, anything else is legacy):

1. **Legacy (linear)** — `depression_scid_example.yaml`: the fixed question
   order, `allow_followup` per question. Loads through the v2 adapter
   (same question order and behavior); the adapter generates
   v2-convention ids (`sec_<n>`, `qg_<n>.1`, `sq_<n>.1.<m>`) and remaps
   the legacy question ids.
2. **v2** — `depression_scid_v2_example.yaml`: the target format of the
   reconstruction (RECONSTRUCTION_PLAN.md Phase 2; authoritative design
   note: `docs/interview_navi_design.md`): a structured interview as a
   navigable object — a hierarchy of Section Groups (arbitrarily
   nestable) > Question Groups (freely configurable internal logic;
   markable as a criteria group) > Single Questions (ID, question text,
   answer type yes/no / number / string, required semantic units), with
   declared navigation `values`, criteria (a criteria group's criterion
   refers to ALL elements of the group together; operators: logical
   NOT/AND/OR, arithmetic equal/unequal/≥/≤/between/min/max, text
   contains/contains-not, completeness empty/not-empty, group-level
   count/threshold/majority/weighted sum), section `relations`
   (temporal / pathological / custom) and `jumps` (direct ID references
   to any question, question group or section group; loops are allowed —
   dead ends are rejected at import). Default navigation is the postorder
   traversal; criteria groups allow deviations, cross-links and loops.
   ⚠️ Status (2026-10-22): the loader/model (Phase 2 step 2.1) now
   implements the full hierarchy — nested sections, `sec_`/`qg_`/`sq_`
   id prefixes (prefix mandatory, numeric part informational),
   `answer_type`, criteria on criteria groups (question group OR section
   group), section relations, and loop-tolerant import validation (only
   dead ends / invalid references are rejected). The Navigator itself
   (step 2.5) is not implemented yet; the full operator set beyond
   `all / any / not / eq / gte / lte / threshold` and the loop-back-
   to-locked-group behavior are open points for that step.

Copy the example of the format you want and edit the copy to create your
own guide. See the comments at the top of each file for what every field
means.

Which guide is currently active is set in `study_content/study_settings.yaml`
under `interview_guide`. Note: the legacy pipeline (`backend/app.py`) can
only present legacy guides; v2 guides are consumed by the new pipeline
(Phase 2 step 2.5 onward / Phase 5).

**How an interview ends (new pipeline, `backend/interview_run.py`):**
there is no declarative end condition in the file — the interview ends
when the Navigator decides nothing more is to be asked and the
Supervisor applies the explicit run-level transition `complete_run`
(every required question is CLOSED or SKIPPED, after which the
`closing_message` is delivered); `abort_run` covers participant abort /
safety early stop. SKIPPING a question is not an ending: it only marks
the question as never-to-be-delivered because the participant already
gave us that information — its criteria can still be evaluated, and the
interview continues. (Legacy pipeline: the interview ends when the fixed
question list is exhausted — `InterviewGuide.is_finished()` — and the
`closing_message` is delivered.)
