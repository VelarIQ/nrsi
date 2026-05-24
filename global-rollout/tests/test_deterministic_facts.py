from nrsip.nrs_core import NRSProcessingPipeline, ProcessingMode


def test_known_factual_query_uses_ground_truth_fact_route():
    """Verify pipeline processes factual queries in DETERMINISTIC mode."""
    pipe = NRSProcessingPipeline()
    pipe._web_engine = None
    result = pipe.process("Who holds the 100m world record?", domain="general")

    assert result.mode == ProcessingMode.DETERMINISTIC
    assert len(result.answer) > 10, "Expected a substantive answer"
    assert result.tier in ("T1", "T2"), f"Expected low complexity tier, got {result.tier}"
    assert result.confidence > 0.0
