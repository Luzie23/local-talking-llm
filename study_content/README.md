# study_content/

This folder holds everything you are likely to edit as the researcher:
question wording, order, prompts, safety keywords, study settings, and
media assets. Everything in here is plain text, YAML, or Markdown — no
programming needed.

You should not need to open anything outside this folder for day-to-day
changes to the study content. Technical code lives in `backend/` and
`frontend/` instead — with one known exception, noted at the bottom of
this file.

## What's here

- **`interviews/`** — the interview guides: all questions, in order, plus
  the framing message shown at the start and the closing message shown at
  the end. `depression_scid_example.yaml` is a working example; copy it to
  create your own guide. Which guide is active is set in
  `study_settings.yaml` (`interview_guide`). See `interviews/README.md`.

- **`prompts/`** — the behavior instructions given to the language model
  (how it should phrase feedback, whether it may ask one follow-up
  question), plus `profiles.yaml`, which names each profile and links it to
  its instructions file. Two profiles exist: `strict_clinical` (exact
  wording, for standardized instruments) and `flexible_conversational`
  (warmer tone, smoothly rewords the transition to the next question).
  Which profile a guide uses is set by that guide's `behavior_profile`
  field. See `prompts/README.md`.

- **`study_settings.yaml`** — small overall settings: interview language,
  which interview guide is active, which TTS voice is used.

- **`safety_keywords.yaml`** — a plain, transparent keyword list (one per
  language, e.g. `de`/`en`) checked against the *participant's own words
  only* (never the model's output). If a keyword matches, the interface
  shows a calm, non-blocking hint with support resources — the interview
  is never interrupted or stopped. Deliberately simple substring matching
  rather than an AI judgment call; see the extensive comments at the top of
  the file for the reasoning, and for the specific lesson that short
  fragments ("nicht mehr da") survive grammatical variation far better than
  full phrases ("nicht mehr da sein"). Freely add, remove, or edit entries
  for your study and language.

- **`safety_keywords_sources.md`** — documents which published literature
  (if any) informed a given addition to `safety_keywords.yaml`, what was
  and wasn't incorporated from each source, and why. Add an entry here
  whenever you extend the keyword list based on a paper or guideline, so
  the reasoning behind the list stays traceable later — not just that an
  entry exists, but where it came from.

- **`assets/`** — images/videos you supply yourself, e.g. a logo for the
  "logo" embodiment condition, or a picture/video for the "picture"/"video"
  conditions. Files here are served at `/study-assets/<filename>` (see
  `backend/app.py`) — set `logoSrc`/`pictureSrc`/`videoSrc` in
  `frontend/conversation.js` accordingly (e.g.
  `logoSrc: "/study-assets/logo.png"`). **Always add an entry to
  `assets/SOURCES.md`** for any new file — especially AI-generated images —
  documenting how it was created (tool, prompt, date), so a replacement can
  be produced the same way later without guessing.

## Not part of the pilot yet

Experimental condition assignment (`conditions.yaml`, `framing.yaml`) —
added once the study moves beyond the fixed-condition pilot (see the
project plan, Milestone 2).

## One known exception to "everything researcher-facing lives here"

A fair amount of researcher-editable text — the start-overlay checklist,
the abort/exit-flow wording, support resources, embodiment file paths
(`logoSrc`/`pictureSrc`/`videoSrc`) — currently lives in
`frontend/conversation.js` instead of in this folder, for technical
reasons (it needs to reach the browser directly). Moving it into
`study_content/` as well is a planned but not-yet-done step — see the
project plan's "Open items" section. Until then, treat
`frontend/conversation.js`'s top section the same way you'd treat a file
in here: it's meant to be edited by you, even though it isn't physically
in this folder.
