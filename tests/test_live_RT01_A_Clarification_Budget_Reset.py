from tests.test_interview_flow import FlowTestBase


class TestLiveRT01ClarificationBudget(FlowTestBase):
    """
    RT01: Clarification requests must not consume the budget.
    The first off-topic answer should trigger a redirection.
    The second consecutive off-topic answer may exhaust the budget
    and advance the interview.
    """

    GENERIC_NEXT = "Danke. Wir gehen zur nächsten Frage weiter."

    def test_clarifications_do_not_consume_budget_and_off_topic_is_limited(self):
        # ---------------------------------------------------------
        # Step 1: First clarification request
        # ---------------------------------------------------------
        response1 = self.say("Ich verstehe die Frage nicht.")
        item1 = self.current()

        assert response1.get("response", "").strip()
        assert self.current() == item1
        assert self.GENERIC_NEXT not in response1["response"]

        # ---------------------------------------------------------
        # Step 2: Second clarification request
        # Clarification/rephrasing must not consume the budget.
        # ---------------------------------------------------------
        response2 = self.say("Ich verstehe die Frage immer noch nicht.")
        
        assert response2.get("response", "").strip()
        assert self.current() == item1
        assert self.GENERIC_NEXT not in response2["response"]

        # ---------------------------------------------------------
        # Step 3: First off-topic answer
        # The chatbot should redirect the participant to the topic.
        # It must not skip the current item yet.
        # ---------------------------------------------------------
        response3 = self.say("Ich habe Hotdogs gegessen.")

        assert response3.get("response", "").strip()
        assert self.current() == item1, (
            "The first off-topic answer must not skip the current item."
        )
        assert self.GENERIC_NEXT not in response3["response"], (
            "The chatbot must not immediately say 'Danke, nächste Frage' "
            "after the first off-topic answer."
        )

        # ---------------------------------------------------------
        # Step 4: Second consecutive off-topic answer
        # The off-topic allowance is now exhausted; the interview
        # may move to the next item.
        # ---------------------------------------------------------
        response4 = self.say("Heute habe ich Pizza gegessen.")

        assert response4.get("response", "").strip()
        assert self.current() != item1, (
            "After the second consecutive off-topic answer, "
            "the current item should be skipped."
        )
