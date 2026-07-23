# study_content/

This folder holds everything you are likely to edit as the researcher:
question wording, order, prompts, and study settings. Everything in here is
plain text or YAML — no programming needed.

You should not need to open anything outside this folder for day-to-day
changes to the study content. Technical code lives in `backend/` and
`frontend/` instead.

## What's here (and what's coming)

- `interviews/` — the interview guides: all questions, in order (e.g. a
  depression screening guide, a general mental-health guide). **Coming in
  the next step** — currently empty.
- `prompts/` — the behavior instructions given to the language model (e.g.
  "ask only the pre-written question", "give a short, neutral, empathetic
  reply"). **Coming in a later step** — currently empty.
- `study_settings.yaml` — small overall settings such as the interview
  language.

Not part of the pilot yet, and therefore not in this folder yet: experimental
condition assignment (`conditions.yaml`, `framing.yaml`) and embodiment
files (logo/photo/video). Those are added once the study moves beyond the
fixed-condition pilot (see the project plan, Milestone 2).
