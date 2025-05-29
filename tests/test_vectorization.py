"""
Comprehensive test suite for the Text Vectorization Agent.

Tests cover core algorithms, preprocessing, API endpoints, and integration scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.algorithms.skip_patterns import PrimeSkipPattern, LogarithmicSkipPattern, sieve_of_eratosthenes
from utilities.data_processing.preprocessing import TextPreprocessor, PreprocessingConfig
from agents.vectorization import VectorizationAgent, VectorConfig


class TestSkipPatterns:
    """Test suite for skip pattern algorithms."""
    
    def test_sieve_of_eratosthenes(self):
        """Test prime number generation."""
        primes = sieve_of_eratosthenes(20)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert primes == expected
    
    def test_sieve_edge_cases(self):
        """Test edge cases for prime generation."""
        assert sieve_of_eratosthenes(1) == []
        assert sieve_of_eratosthenes(2) == [2]
        assert sieve_of_eratosthenes(0) == []
    
    def test_prime_skip_pattern_initialization(self):
        """Test PrimeSkipPattern initialization."""
        pattern = PrimeSkipPattern(max_distance=10)
        assert pattern.max_distance == 10
        assert 2 in pattern.primes
        assert 3 in pattern.primes
        assert 5 in pattern.primes
        assert 7 in pattern.primes
    
    def test_prime_skip_pattern_extraction(self):
        """Test prime pattern sequence extraction."""
        pattern = PrimeSkipPattern(max_distance=10)
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        
        sequences = pattern.extract_sequences(tokens)
        assert len(sequences) > 0
        
        # Each sequence should have the target word as first element
        for seq in sequences:
            assert seq[0] in tokens
    
    def test_prime_context_pairs(self):
        """Test prime pattern context pair extraction."""
        pattern = PrimeSkipPattern(max_distance=5)
        tokens = ["a", "b", "c", "d", "e", "f"]
        
        pairs = pattern.extract_context_pairs(tokens)
        assert len(pairs) > 0
        
        # All pairs should be tuples of strings
        for target, context in pairs:
            assert isinstance(target, str)
            assert isinstance(context, str)
            assert target in tokens
            assert context in tokens
    
    def test_logarithmic_skip_pattern(self):
        """Test logarithmic pattern sequence extraction."""
        pattern = LogarithmicSkipPattern(max_distance=5)
        tokens = ["word1", "word2", "word3", "word4", "word5", "word6"]
        
        # Set random seed for reproducible results
        np.random.seed(42)
        sequences = pattern.extract_sequences(tokens)
        
        assert len(sequences) > 0
        for seq in sequences:
            assert len(seq) >= 1  # At least contains the target word
    
    def test_log_weight_calculation(self):
        """Test logarithmic weight calculation."""
        pattern = LogarithmicSkipPattern()
        
        # Weight should decrease with distance
        weight1 = pattern._log_weight(1)
        weight2 = pattern._log_weight(2)
        weight5 = pattern._log_weight(5)
        
        assert weight1 > weight2 > weight5
        assert all(w > 0 for w in [weight1, weight2, weight5])
    
    def test_weighted_context_extraction(self):
        """Test weighted context extraction."""
        pattern = LogarithmicSkipPattern(max_distance=3)
        tokens = ["center", "word1", "word2", "word3"]
        
        weighted_context = pattern.get_weighted_context(tokens, center_idx=0)
        
        # Should return (word, weight) tuples
        assert len(weighted_context) == 3
        for word, weight in weighted_context:
            assert isinstance(word, str)
            assert isinstance(weight, float)
            assert weight > 0


class TestTextPreprocessor:
    """Test suite for text preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a text preprocessor for testing."""
        config = PreprocessingConfig(
            remove_stopwords=True,
            remove_punctuation=True,
            lowercase=True,
            lemmatize=False  # Disable to avoid spaCy model dependency in tests
        )
        return TextPreprocessor(config=config)
    
    def test_initial_clean(self, preprocessor):
        """Test initial text cleaning."""
        dirty_text = "Visit   https://example.com for more info!!! Email us at test@example.com..."
        cleaned = preprocessor._initial_clean(dirty_text)
        
        assert "https://example.com" not in cleaned
        assert "test@example.com" not in cleaned
        assert "!!!" not in cleaned
        assert "..." not in cleaned
    
    def test_empty_text_handling(self, preprocessor):
        """Test handling of empty or whitespace-only text."""
        assert preprocessor.process("") == []
        assert preprocessor.process("   ") == []
        assert preprocessor.process("\n\t") == []
    
    def test_batch_processing(self, preprocessor):
        """Test batch text processing."""
        texts = [
            "This is the first document.",
            "This is the second document.",
            ""  # Empty text
        ]
        
        results = preprocessor.process_batch(texts)
        assert len(results) == 3
        assert len(results[2]) == 0  # Empty text should return empty list


class TestVectorizationAgent:
    """Test suite for the main vectorization agent."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return VectorConfig(
            vector_size=50,  # Smaller for testing
            window=3,
            min_count=1,
            epochs=1,  # Single epoch for speed
            use_prime_patterns=True,
            use_log_patterns=True
        )
    
    @pytest.fixture
    def agent(self, config):
        """Create a vectorization agent for testing."""
        return VectorizationAgent(config=config)
    
    def test_agent_initialization(self, agent, config):
        """Test agent initialization."""
        assert agent.config == config
        assert agent.model is None
        assert agent.vocabulary is None
    
    def test_training_with_sample_data(self, agent):
        """Test training with sample text data."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question."
        ]
        
        model = agent.train(texts)
        
        assert model is not None
        assert agent.model is not None
        assert agent.vocabulary is not None
        assert len(agent.vocabulary) > 0
    
    def test_vector_retrieval(self, agent):
        """Test vector retrieval after training."""
        texts = ["The cat sat on the mat.", "The dog ran in the park."]
        agent.train(texts)
        
        # Try to get vectors for common words
        vector = agent.get_vector("the")
        if vector is not None:
            assert isinstance(vector, np.ndarray)
            assert len(vector) == agent.config.vector_size
    
    def test_similarity_calculation(self, agent):
        """Test similarity calculation between words."""
        texts = ["good bad", "love hate", "big small"] * 10  # Repeat for frequency
        agent.train(texts)
        
        # Test similarity if words exist in vocabulary
        similarity = agent.calculate_similarity("good", "bad")
        if similarity is not None:
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1
    
    def test_similar_words_search(self, agent):
        """Test finding similar words."""
        texts = ["good great excellent", "bad terrible awful"] * 5
        agent.train(texts)
        
        similar = agent.find_similar("good", topn=2)
        assert isinstance(similar, list)
        # If similar words found, they should be tuples of (word, score)
        for word, score in similar:
            assert isinstance(word, str)
            assert isinstance(score, float)
    
    def test_model_info(self, agent):
        """Test model information retrieval."""
        # Before training
        info = agent.get_model_info()
        assert info == {}
        
        # After training
        texts = ["test sentence for model info"]
        agent.train(texts)
        
        info = agent.get_model_info()
        assert "vocabulary_size" in info
        assert "vector_size" in info
        assert info["vector_size"] == agent.config.vector_size
    
    def test_model_save_load(self, agent):
        """Test model saving and loading."""
        texts = ["sample text for save load test"]
        agent.train(texts)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.model")
            
            # Save model
            agent._save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new agent and load model
            new_agent = VectorizationAgent(agent.config)
            new_agent.load_model(model_path)
            
            assert new_agent.model is not None
            assert new_agent.vocabulary is not None


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from text to vectors."""
        # Sample text
        text = """
        The artificial intelligence system processes natural language
        to create semantic vector representations. Machine learning
        algorithms analyze textual patterns and relationships.
        """
        
        # Configuration
        config = VectorConfig(
            vector_size=100,
            window=5,
            min_count=1,
            epochs=2,
            use_prime_patterns=True,
            use_log_patterns=True
        )
        
        # Initialize components
        agent = VectorizationAgent(config=config)
        preprocessor = TextPreprocessor()
        
        # Process and train
        processed_tokens = preprocessor.process(text)
        assert len(processed_tokens) > 0
        
        model = agent.train([text])
        assert model is not None
        
        # Test semantic operations
        if "system" in agent.vocabulary and "algorithm" in agent.vocabulary:
            similarity = agent.calculate_similarity("system", "algorithm")
            assert similarity is not None
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        agent = VectorizationAgent()
        
        # Test operations on untrained model
        assert agent.get_vector("test") is None
        assert agent.find_similar("test") == []
        assert agent.calculate_similarity("word1", "word2") is None
        
        # Test with empty training data
        with pytest.raises(ValueError):
            agent.train([])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
