
from backend.agents.navigation_agent import decide_next_action

def check(label, **kwargs):
    result = decide_next_action(**kwargs)
    print(f"{label}: action={result.action}, "
          f"mode={result.question_mode}, "
          f"skip_reason={result.skip_reason}")
    return result

common = {
    "sufficiency": "ambiguous",
    "clarification_count": 0,
    "max_clarifications": 1,
    "next_step": None,
    "cannot_answer_moves_on": False,
}

# 1. Katılımcı sorunun neden sorulduğunu soruyor
r1 = check(
    "Clarification request",
    intent="clarification_request",
    **common,
)
assert r1.action == "rephrase"

# 2. Katılımcı soruyu anlamadığını söylüyor,
# önceki açıklama bütçesi dolmuş olsa bile
r2 = check(
    "Still doesn't understand",
    intent="clarification_request",
    sufficiency="insufficient",
    clarification_count=1,
    max_clarifications=1,
    next_step=None,
    cannot_answer_moves_on=False,
)
assert r2.action == "rephrase"

# 3. Katılımcı yanıt vermeyi reddediyor
r3 = check(
    "First refusal",
    intent="refusal",
    **common,
)
assert r3.action == "clarify"

# 4. Katılımcı cevap veremiyor ve bütçe dolmuş
r4 = check(
    "Cannot answer after limit",
    intent="cannot_answer",
    sufficiency="insufficient",
    clarification_count=1,
    max_clarifications=1,
    next_step=None,
    cannot_answer_moves_on=False,
)
assert r4.action == "skip"

# 5. Yanıt yeterli
r5 = check(
    "Sufficient answer",
    intent="answer",
    sufficiency="sufficient",
    **{k: v for k, v in common.items() if k != "sufficiency"},
)
assert r5.action == "advance"

print("\nAll navigation checks passed.")
