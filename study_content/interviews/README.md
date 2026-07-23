# interviews/

This folder holds the interview guides — one YAML file per guide (for
example a depression screening guide, or a general mental-health guide).
Each file lists the fixed questions in order, exactly as the assistant
presents them to the participant, plus the framing message shown at the
start and the closing message shown at the end.

`depression_scid_example.yaml` is a working example — copy it and edit the
copy to create your own guide. See the comments at the top of that file for
what each field means.

Which guide is currently active is set in `study_content/study_settings.yaml`
under `interview_guide`.

