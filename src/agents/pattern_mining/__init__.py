"""Pattern Mining Agent

Discovers emergent linguistic patterns and thematic trends in text corpora.
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..utilities.data_processing.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PatternMiningConfig:
    """Configuration for pattern mining operations."""
    ngram_min: int = 2
    ngram_max: int = 3
    top_k: int = 10


class PatternMiningAgent:
    """Agent for discovering linguistic patterns across text collections."""

    def __init__(self, config: Optional[PatternMiningConfig] = None) -> None:
        self.config = config or PatternMiningConfig()
        self.preprocessor = TextPreprocessor()
        logger.info("PatternMiningAgent initialized")

    def _generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def extract_ngram_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Extract n-gram frequencies from a list of texts."""
        if not texts:
            return {}

        ngram_counter: Counter[str] = Counter()
        for text in texts:
            tokens = self.preprocessor.process(text)
            for n in range(self.config.ngram_min, self.config.ngram_max + 1):
                ngrams = self._generate_ngrams(tokens, n)
                ngram_counter.update(ngrams)

        return dict(ngram_counter.most_common(self.config.top_k))

    def segment_texts(self, texts: List[str], segment_size: int) -> List[List[str]]:
        """Split texts into sequential segments of fixed size."""
        segments: List[List[str]] = []
        current: List[str] = []
        for text in texts:
            current.append(text)
            if len(current) >= segment_size:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments

    def trending_terms(self, texts: List[str], segment_size: int = 10) -> List[Tuple[int, Dict[str, int]]]:
        """Analyze term frequency trends across sequential segments."""
        segments = self.segment_texts(texts, segment_size)
        trends: List[Tuple[int, Dict[str, int]]] = []
        for idx, segment in enumerate(segments):
            freq = self.extract_ngram_frequencies(segment)
            trends.append((idx, freq))
        return trends


__all__ = ["PatternMiningAgent", "PatternMiningConfig"]
