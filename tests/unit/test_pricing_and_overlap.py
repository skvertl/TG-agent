import pytest
from core.observability.pricing import calculate_cost, get_model_rates
from core.observability.overlap import estimate_tokens, calculate_context_overlap


class TestPricing:
    def test_qwen_pricing(self):
        cost = calculate_cost(
            model="qwen2.5:7b",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached_tokens=0,
        )
        assert cost == pytest.approx(1.50, rel=1e-3)

    def test_llama_pricing(self):
        cost = calculate_cost(
            model="llama3.2:latest",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached_tokens=0,
        )
        assert cost == pytest.approx(0.75, rel=1e-3)

    def test_cached_tokens_discount(self):
        cost_normal = calculate_cost(
            model="qwen2.5:7b",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=0,
        )
        cost_cached = calculate_cost(
            model="qwen2.5:7b",
            input_tokens=0,
            output_tokens=0,
            cached_tokens=1_000_000,
        )
        assert cost_cached < cost_normal
        assert cost_cached == pytest.approx(0.075, rel=1e-3)

    def test_fallback_model_rates(self):
        rates = get_model_rates("unknown-custom-model")
        assert rates["input_per_1m"] > 0
        assert rates["output_per_1m"] > 0


class TestOverlap:
    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("   ") == 0

    def test_estimate_tokens_length(self):
        text = "Hello world! This is a simple sentence for testing tokens."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 15

    def test_calculate_context_overlap_first_turn(self):
        prev = []
        curr = ["System prompt", "User prompt"]
        repeated_tokens, pct = calculate_context_overlap(prev, curr)
        assert repeated_tokens == 0
        assert pct == 0.0

    def test_calculate_context_overlap_second_turn(self):
        prev = ["System prompt", "User prompt"]
        curr = ["System prompt", "User prompt", "Assistant: Hello", "User: Next question"]
        repeated_tokens, pct = calculate_context_overlap(prev, curr)
        assert repeated_tokens > 0
        assert 30.0 <= pct <= 80.0
