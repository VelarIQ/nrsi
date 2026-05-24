from nrsip.client_response_policy import apply_client_response_policy


def test_client_response_policy_allows_high_confidence_output():
    decision = apply_client_response_policy(
        answer="safe answer",
        observed_confidence=0.91,
        floor=0.80,
    )

    assert decision.client_visible is True
    assert decision.answer == "safe answer"
    assert decision.delivered_confidence == 0.91
    assert decision.reason is None


def test_client_response_policy_withholds_low_confidence_output():
    decision = apply_client_response_policy(
        answer="unsafe to show",
        observed_confidence=0.58,
        floor=0.80,
    )

    assert decision.client_visible is False
    assert decision.answer.startswith("I can't return a reliable answer")
    assert decision.delivered_confidence == 0.0
    assert decision.observed_confidence == 0.58
    assert "below floor 0.800" in (decision.reason or "")
