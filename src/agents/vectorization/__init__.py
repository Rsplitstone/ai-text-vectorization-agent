"""
Vectorization Agent - Core Implementation

Implements advanced skip-gram methodologies with novel pattern recognition algorithms
for semantic vector space construction.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import spacy
from gensim.models import Word2Vec
from ..core.algorithms.skip_patterns import PrimeSkipPattern, LogarithmicSkipPattern
from ..utilities.data_processing.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """Configuration for vector training parameters."""
    vector_size: int = 300
    window: int = 5
    min_count: int = 5
    workers: int = 4
    epochs: int = 10
    alpha: float = 0.025
    min_alpha: float = 0.0001
    use_prime_patterns: bool = True
    use_log_patterns: bool = True
    max_prime_distance: int = 50


class VectorizationAgent:
    """
    Advanced text vectorization agent implementing novel skip-gram patterns.
    
    Features:
    - Prime-number based context extraction
    - Logarithmic weighting schemes
    - Multidimensional semantic vector spaces
    - Scalable processing pipeline
    """
    
    def __init__(self, config: Optional[VectorConfig] = None):
        """Initialize the vectorization agent with configuration."""
        self.config = config or VectorConfig()
        self.preprocessor = TextPreprocessor()
        self.prime_pattern = PrimeSkipPattern(max_distance=self.config.max_prime_distance)
        self.log_pattern = LogarithmicSkipPattern()
        self.model: Optional[Word2Vec] = None
        self.vocabulary: Optional[Dict[str, int]] = None
        
        logger.info(f"VectorizationAgent initialized with config: {self.config}")
    
    def train(self, texts: List[str], model_path: Optional[str] = None) -> Word2Vec:
        """
        Train the vectorization model on input texts.
        
        Args:
            texts: List of input text documents
            model_path: Optional path to save the trained model
            
        Returns:
            Trained Word2Vec model
        """
        logger.info(f"Starting training on {len(texts)} documents")
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            tokens = self.preprocessor.process(text)
            if len(tokens) > 10:  # Filter very short documents
                processed_texts.append(tokens)
        
        logger.info(f"Preprocessed {len(processed_texts)} valid documents")
        
        # Generate enhanced training data with novel skip patterns
        training_data = self._generate_training_data(processed_texts)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=training_data,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.workers,
            epochs=self.config.epochs,
            alpha=self.config.alpha,
            min_alpha=self.config.min_alpha,
            sg=1  # Skip-gram model
        )
        
        # Build vocabulary mapping
        self.vocabulary = {word: self.model.wv.key_to_index[word] 
                          for word in self.model.wv.key_to_index}
        
        logger.info(f"Training completed. Vocabulary size: {len(self.vocabulary)}")
        
        # Save model if path provided
        if model_path:
            self._save_model(model_path)
            
        return self.model
    
    def _generate_training_data(self, processed_texts: List[List[str]]) -> List[List[str]]:
        """Generate enhanced training data using novel skip patterns."""
        training_data = []
        
        for tokens in processed_texts:
            # Standard sequential processing
            training_data.append(tokens)
            
            # Add prime-pattern enhanced sequences
            if self.config.use_prime_patterns:
                prime_sequences = self.prime_pattern.extract_sequences(tokens)
                training_data.extend(prime_sequences)
            
            # Add logarithmic-pattern enhanced sequences  
            if self.config.use_log_patterns:
                log_sequences = self.log_pattern.extract_sequences(tokens)
                training_data.extend(log_sequences)
        
        logger.info(f"Generated {len(training_data)} training sequences")
        return training_data
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector representation for a word."""
        if not self.model or word not in self.model.wv:
            return None
        return self.model.wv[word]
    
    def find_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words to the given word."""
        if not self.model or word not in self.model.wv:
            return []
        return self.model.wv.most_similar(word, topn=topn)
    
    def calculate_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Calculate cosine similarity between two words."""
        if not self.model or word1 not in self.model.wv or word2 not in self.model.wv:
            return None
        return self.model.wv.similarity(word1, word2)
    
    def _save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a pre-trained model from disk."""
        self.model = Word2Vec.load(path)
        self.vocabulary = {word: self.model.wv.key_to_index[word] 
                          for word in self.model.wv.key_to_index}
        logger.info(f"Model loaded from {path}. Vocabulary size: {len(self.vocabulary)}")
    
    def get_model_info(self) -> Dict[str, Union[int, float]]:
        """Get information about the trained model."""
        if not self.model:
            return {}
        
        return {
            "vocabulary_size": len(self.vocabulary) if self.vocabulary else 0,
            "vector_size": self.model.wv.vector_size,
            "total_words": self.model.corpus_total_words,
            "epochs": self.model.epochs
        }
