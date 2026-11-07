
"""RT06–RT07: Interview termination, refusal, and explicit skip."""

import os
import sys

BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend")
)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from test_interview_flow import FlowTestBase, SCRIPT, Q2


STOP_REQUEST = "Ich möchte das Gespräch beenden."
ITEM_REFUSAL = "Diese Frage möchte ich nicht beantworten."
EXPLICIT_SKIP = "Bitte überspringe diese Frage."


def print_report(name, answer, response, before, after, finished, records):
    print("\n" + "=" * 85)
    print(f"TEST: {name}")
    print(f"INPUT: {answer!r}")
    print(f"RESPONSE: {response!r}")
    print(f"CURRENT BEFORE: {before!r}")
    print(f"CURRENT AFTER: {after!r}")
    print(f"FINISHED: {finished!r}")
    print(f"RECORDS: {records!r}")
    print("=" * 85)


class TestRT06RT07RefusalSkipAndTermination(FlowTestBase):
    """Tests for the intended termination, refusal, and skip behavior."""

    def setUp(self):
        super().setUp()
        self.original_script = SCRIPT.copy()

        # The end_interview intent is an intended future capability.
        # Production support will be implemented after the tests are fixed.
        SCRIPT[STOP_REQUEST] = (
            "end_interview", "insufficient", [], "", None
        )
        SCRIPT[ITEM_REFUSAL] = (
            "refusal", "insufficient", [], "", None
        )

    def tearDown(self):
        try:
            SCRIPT.clear()
            SCRIPT.update(self.original_script)
        finally:
            super().tearDown()

    def test_rt06a_stop_request_ends_the_entire_interview(self):
        """RT06-A: A stop request terminates the whole interview."""

        before = self.current()
        result = self.say(STOP_REQUEST)

        response = result.get("response", "")
        after = self.current()
        finished = result.get("finished", False)
        records = self.records()

        print_report(
            "RT06-A | Stop entire interview",
            STOP_REQUEST,
            response,
            before,
            after,
            finished,
            records,
        )

        assert before == "dep_01"
        assert finished is True, (
            "RT06-A FAIL: A stop request must set finished=True."
        )
        assert after is None, (
            "RT06-A FAIL: No active interview item should remain."
        )
        assert Q2 not in response, (
            "RT06-A FAIL: The next question must not be asked after termination."
        )

    def test_rt06b_item_refusal_skips_only_the_current_item(self):
        """RT06-B: Refusing one question must not terminate the interview."""

        before = self.current()
        result = self.say(ITEM_REFUSAL)

        response = result.get("response", "")
        after = self.current()
        finished = result.get("finished", False)
        records = self.records()

        print_report(
            "RT06-B | Refuse current item only",
            ITEM_REFUSAL,
            response,
            before,
            after,
            finished,
            records,
        )

        assert before == "dep_01"
        assert finished is False, (
            "RT06-B FAIL: Refusing one item must not end the interview."
        )
        assert after == "dep_02", (
            "RT06-B FAIL: The interview should advance to the next item."
        )
        assert Q2 in response, (
            "RT06-B FAIL: The next question should be presented."
        )
        assert len(records) == 1
        assert records[0]["status"] == "skipped"
        assert records[0]["skip_reason"] == "refused"

    def test_rt07_explicit_skip_skips_current_item_without_judgment_agent(self):
        """RT07: An explicit skip command bypasses the Judgment Agent."""

        before = self.current()
        result = self.say(EXPLICIT_SKIP)

        response = result.get("response", "")
        after = self.current()
        finished = result.get("finished", False)
        records = self.records()

        print_report(
            "RT07 | Explicit skip command",
            EXPLICIT_SKIP,
            response,
            before,
            after,
            finished,
            records,
        )

        assert before == "dep_01"
        assert after == "dep_02", (
            "RT07 FAIL: The explicit skip command must skip the current item."
        )
        assert finished is False, (
            "RT07 FAIL: Skipping one item must not terminate the interview."
        )
        assert Q2 in response, (
            "RT07 FAIL: The next interview question should be displayed."
        )
        assert len(self.model.judgment_prompts) == 0, (
            "RT07 FAIL: Explicit skip must be handled without the Judgment Agent."
        )
        assert len(records) == 1
        assert records[0]["status"] == "skipped"
        assert records[0]["skip_reason"] == "refused"


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(["-v", "-s", os.path.abspath(__file__)])
    )
