"""
Text Preprocessing Pipeline

Advanced text preprocessing utilities for the vectorization agent.
Includes spaCy integration and custom processing steps.
"""

import re
import spacy
from typing import List, Optional, Set, Dict
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing parameters."""
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    min_token_length: int = 2
    max_token_length: int = 50
    remove_numbers: bool = False
    lemmatize: bool = True
    remove_entities: bool = False
    custom_stopwords: Optional[Set[str]] = None


class TextPreprocessor:
    """
    Advanced text preprocessing pipeline using spaCy.
    
    Features:
    - Intelligent tokenization and normalization
    - Named entity recognition and filtering
    - Custom stopword management
    - Lemmatization and stemming
    - Pattern-based text cleaning
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor with spaCy model and configuration."""
        self.config = config or PreprocessingConfig()
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Disable unnecessary pipeline components for performance
        self.nlp.disable_pipes(["parser", "ner", "textcat"])
        if not self.config.remove_entities:
            self.nlp.enable_pipe("ner")
        
        # Custom stopwords
        self.custom_stopwords = self.config.custom_stopwords or set()
        
        logger.info("TextPreprocessor initialized successfully")
    
    def process(self, text: str) -> List[str]:
        """
        Process a single text document and return clean tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        if not text or not text.strip():
            return []
        
        # Initial cleaning
        cleaned_text = self._initial_clean(text)
        
        # Process with spaCy
        doc = self.nlp(cleaned_text)
        
        # Extract and filter tokens
        tokens = []
        for token in doc:
            processed_token = self._process_token(token)
            if processed_token:
                tokens.append(processed_token)
        
        return tokens
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process multiple texts efficiently using spaCy's batch processing.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of token lists for each text
        """
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [self._initial_clean(text) for text in texts]
        
        # Batch process with spaCy
        processed_texts = []
        for doc in self.nlp.pipe(cleaned_texts, batch_size=50):
            tokens = []
            for token in doc:
                processed_token = self._process_token(token)
                if processed_token:
                    tokens.append(processed_token)
            processed_texts.append(tokens)
        
        logger.info(f"Processed {len(texts)} texts in batch")
        return processed_texts
    
    def _initial_clean(self, text: str) -> str:
        """Perform initial text cleaning before spaCy processing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _process_token(self, token) -> Optional[str]:
        """Process a single spaCy token and apply filtering rules."""
        # Skip if token doesn't exist
        if not token.text or not token.text.strip():
            return None
        
        # Get token text
        text = token.lemma_ if self.config.lemmatize else token.text
        
        # Apply case transformation
        if self.config.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.config.remove_punctuation and token.is_punct:
            return None
        
        # Remove stopwords
        if self.config.remove_stopwords and (token.is_stop or text in self.custom_stopwords):
            return None
        
        # Remove numbers
        if self.config.remove_numbers and (token.like_num or text.isdigit()):
            return None
        
        # Length filtering
        if len(text) < self.config.min_token_length or len(text) > self.config.max_token_length:
            return None
        
        # Remove tokens with only special characters
        if re.match(r'^[^a-zA-Z0-9]+$', text):
            return None
        
        # Remove entities if configured
        if self.config.remove_entities and token.ent_type_:
            return None
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        if "ner" not in self.nlp.pipe_names:
            self.nlp.enable_pipe("ner")
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(ent.text)
        
        return entities
    
    def get_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Get preprocessing statistics for a collection of texts."""
        if not texts:
            return {}
        
        processed_texts = self.process_batch(texts)
        
        total_original_tokens = sum(len(text.split()) for text in texts)
        total_processed_tokens = sum(len(tokens) for tokens in processed_texts)
        
        avg_original_length = np.mean([len(text.split()) for text in texts])
        avg_processed_length = np.mean([len(tokens) for tokens in processed_texts])
        
        return {
            "total_texts": len(texts),
            "total_original_tokens": total_original_tokens,
            "total_processed_tokens": total_processed_tokens,
            "reduction_ratio": 1 - (total_processed_tokens / total_original_tokens) if total_original_tokens > 0 else 0,
            "avg_original_length": avg_original_length,
            "avg_processed_length": avg_processed_length
        }
    
    def save_vocabulary(self, texts: List[str], output_path: str) -> None:
        """Extract and save vocabulary from processed texts."""
        processed_texts = self.process_batch(texts)
        
        vocabulary = set()
        for tokens in processed_texts:
            vocabulary.update(tokens)
        
        # Save vocabulary
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in sorted(vocabulary):
                f.write(f"{word}\n")
        
        logger.info(f"Vocabulary of {len(vocabulary)} words saved to {output_path}")


# Import numpy for statistics
import numpy as np
