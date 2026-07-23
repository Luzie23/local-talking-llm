# prompts/

This folder holds the plain-text behavior instructions given to the
language model, plus `profiles.yaml`, which names each profile and links
it to its instructions file.

Two profiles exist so far:
- `strict_clinical.txt` — for standardized, validated instruments (e.g.
  SCID-style modules) where exact wording matters. The next fixed question
  is attached completely unchanged after the model's feedback.
- `flexible_conversational.txt` — for more exploratory questionnaires,
  with a warmer tone. The model also smoothly rewords the transition into
  the next question (without changing its meaning).

Which profile a given interview guide uses is set by that guide's
`behavior_profile` field (see `study_content/interviews/*.yaml`).

Lines starting with "#" in the .txt files are comments for you and are
stripped out before the text is sent to the model — edit the instructions
below them freely, or copy one of the files to create a new profile (and
add a matching entry to `profiles.yaml`).
