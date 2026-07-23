# prompts/

This folder will hold the plain-text behavior instructions given to the
language model — for example, "only ask the pre-written question", "give a
short, neutral, empathetic reply", "never invent your own questions or give
a clinical assessment". These are the rules that keep the assistant
presenting, not improvising — which matters most when talking with patients.

This folder is currently empty. It will be filled in a later planned step
("constrain the LLM") — see the project plan for details. Right now, the
temporary instructions the assistant uses are still written directly inside
`backend/llm_provider.py`, clearly marked as a placeholder.
