# interviews/

This folder holds the interview guides — one YAML file per guide (for
example a depression screening guide, or a general mental-health guide).
Each file lists the fixed questions in order, exactly as the assistant
presents them to the participant, plus the framing message shown at the
start and the closing message shown at the end.

`depression_scid_example.yaml` is a working example — copy it and edit the
copy to create your own guide. See the comments at the top of that file for
what each field means. Note: its questions are team-written development
content, NOT SCID wording.

`scid5pd_de.yaml` (active) is the SKID-5-PD interview (all 10 personality
disorders + Other Specified PD). The German wording is an UNOFFICIAL working
translation of the English SCID-5-PD and must be reviewed by the project
psychologist. See the header of that file for the structure, the items the
chatbot does not ask, and the source-defined routing.

Which guide is currently active is set in `study_content/study_settings.yaml`
under `interview_guide`.

