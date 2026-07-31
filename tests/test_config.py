import pytest

from harness.config import compute_cost


def test_compute_cost_known_token_count():
    # claude-sonnet-5: $2.00 / MTok input, $10.00 / MTok output
    cost = compute_cost("claude-sonnet-5", tokens_in=1000, tokens_out=500)

    expected = 1000 * 2.00 / 1_000_000 + 500 * 10.00 / 1_000_000
    assert cost == pytest.approx(expected)
    assert cost == pytest.approx(0.007)


def test_compute_cost_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        compute_cost("not-a-real-model", tokens_in=100, tokens_out=100)
