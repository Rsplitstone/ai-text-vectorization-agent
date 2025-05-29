"""Tests for PatternMiningAgent."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.pattern_mining import PatternMiningAgent, PatternMiningConfig


def test_ngram_frequencies():
    texts = [
        "the quick brown fox",
        "the quick blue hare",
        "a quick movement of the enemy"
    ]
    agent = PatternMiningAgent(PatternMiningConfig(ngram_min=2, ngram_max=2, top_k=3))
    freqs = agent.extract_ngram_frequencies(texts)
    assert isinstance(freqs, dict)
    assert len(freqs) <= 3
    assert "the quick" in freqs


def test_trending_terms():
    texts = [f"sample text {i}" for i in range(20)]
    agent = PatternMiningAgent()
    trends = agent.trending_terms(texts, segment_size=5)
    assert len(trends) == 4
    for idx, freq in trends:
        assert isinstance(idx, int)
        assert isinstance(freq, dict)
