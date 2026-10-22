# interviews/

This folder holds the interview guides — one file per guide (for example
a depression screening guide, or a general mental-health guide). Each file
lists the fixed questions in order, exactly as the assistant presents them
to the participant, plus the framing message shown at the start and the
closing message shown at the end.

**Two formats are supported** (auto-detected by
`backend/interview_definition.py`: a top-level `groups` key means v2,
anything else is legacy):

1. **Legacy (linear)** — `depression_scid_example.yaml`: the fixed question
   order, `allow_followup` per question. Loads unchanged through the legacy
   adapter (same question order and behavior).
2. **v2** — `manic+hypomanic_scid_v2_example.yaml`: the target format of the
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
   count/threshold/majority/weighted sum), and `jumps` (direct ID
   references to any question, question group or section group; dead
   ends are rejected at import). Default navigation is the postorder
   traversal; criteria groups allow deviations, cross-links and loops.
   ⚠️ Status (2026-10-22): the example and the loader currently implement
   the FLAT MVP subset (one group level, `all_required` (default) /
   `sequential` strategies, the `all / any / not / eq / gte / lte /
   threshold` operators, loop-free import check); the rework to the full
   hierarchical format is the Phase 2 step 2.1 rework.

Copy the example of the format you want and edit the copy to create your
own guide. See the comments at the top of each file for what every field
means.

Which guide is currently active is set in `study_content/study_settings.yaml`
under `interview_guide`. Note: the legacy pipeline (`backend/app.py`) can
only present legacy guides; v2 guides are consumed by the new pipeline
(Phase 2 step 2.5 onward / Phase 5).
