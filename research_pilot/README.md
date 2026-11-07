# First working research experiment

This is an incremental extension of the existing chatbot, inspired by MAGI's
separation of navigation and judgment. It is not a full MAGI reproduction.

## What exists now

- Existing German three-question guide and existing browser interface.
- `--text-only` starts without loading Whisper, Piper, or microphone libraries.
- `--pilot` adds a local Ollama judgment for each answer. It records a finding
  as `present`, `absent`, `unknown`, or `conflicting`, with exact participant quotes.
- The controller asks one fixed clarification for unknown/conflicting answers,
  then advances while preserving unresolved evidence. This clarification policy
  is independent of the original `allow_followup` flags, deliberately a separate condition.
- Report assembly is code, not a diagnosis agent. There is no generated diagnosis.
- A JSON report shows the node, component, transition, raw judgment, validated
  judgment, call time, and tokens. This makes errors inspectable.
- Six German synthetic development cases cover explicit reports, denials,
  missing information, wrong time periods, contradiction, and correction.

## Start locally

Use a Python 3.11+ environment with the repository's text dependencies:

```bash
python -m pip install flask PyYAML langchain-core langchain-ollama
python backend/app.py --web --text-only --pilot --model gemma3:4b --port 8765
```

Ollama must already be running with `gemma3:4b` installed. No paid API is called.
Visit <http://127.0.0.1:8765>. Use fictional answers. The in-memory report is at
<http://127.0.0.1:8765/api/pilot/report>. A page reload resets this single-user
session; save the report before reloading. Omit `--pilot` to run the original
conversation logic in text-only mode. Audio paths were not changed beyond
deferring their imports and returning disabled responses in text-only mode.

Run the development cases:

```bash
python backend/pilot.py --model gemma3:4b --output /tmp/pilot-results.json
python -m unittest tests.test_pilot tests.test_interview_guide tests.test_llm_provider
```

The CLI replays topic-indexed answer fixtures, not simulated adaptive patients.
Expected labels are excluded from model prompts. Its printed agreement counts
are a debugging aid. The six cases and expected labels were drafted together
and need independent human review; they are not a validated or held-out benchmark.

## What the checks establish

An exact quote check can catch fabricated citations or references to the wrong
participant turn. It cannot establish whether a real quote supports the proposed
interpretation. Semantic support and all assessment labels need human evaluation.
This pilot does not establish reduced hallucination, clinical accuracy, or
superiority over a single-agent system. All inference stays on the local Ollama
endpoint. Existing safety code remains outside this experiment's evaluation scope.

## Concrete deliverable for Friday, 23 October 2026

1. Independently review the rubric and six cases with a teammate, recording disagreements.
2. Expand to 20-30 cases, separating development cases from an untouched test set.
3. Add the SAME final assessment rubric to the original chatbot baseline. Compare
   identical transcripts/model/settings before comparing whole adaptive interviews.
4. Record semantic evidence-support errors, finding agreement, uncertainty handling,
   item coverage, model calls, tokens, and latency. Include all failures.
5. Show three traces: correct, uncertain, and failed. Report observed bottlenecks
   and provisional KPI targets. Do not claim model or architecture superiority
   from this small development run.

Only after this evidence should we add model-based questioning, a diagnosis/assessment
agent, or more flexible orchestration. Change one component at a time and keep
the original condition reproducible. Different provider/model results belong to
separate experiments, not the architecture comparison.

## Sources

- MAGI, Sections 3-4: <https://aclanthology.org/2025.findings-acl.1278/>
- Local chat API: <https://docs.ollama.com/api/chat>
- Structured output: <https://docs.ollama.com/capabilities/structured-outputs>

The rubric describes limited observable findings, not licensed MINI wording or
full diagnostic criteria. The three starting questions come from this repository.
