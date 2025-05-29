"""
Training Script for Vectorization Agent

Command-line interface for training the text vectorization model
with advanced skip-gram patterns.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.vectorization import VectorizationAgent, VectorConfig
from utilities.data_processing.preprocessing import TextPreprocessor, PreprocessingConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def load_text_file(file_path: str) -> str:
    """Load text content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise


def load_texts_from_directory(directory: str) -> List[str]:
    """Load all text files from a directory."""
    texts = []
    directory_path = Path(directory)
    
    for file_path in directory_path.glob("*.txt"):
        try:
            text = load_text_file(str(file_path))
            texts.append(text)
            logging.info(f"Loaded {file_path.name}")
        except Exception as e:
            logging.warning(f"Skipping {file_path.name}: {e}")
    
    return texts


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train AI-Driven Text Vectorization Agent"
    )
    
    # Input arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input text file or directory containing text files"
    )
    
    # Output arguments
    parser.add_argument(
        "--model", "-m",
        default="artifacts/vectors.model",
        help="Output path for trained model (default: artifacts/vectors.model)"
    )
    
    # Model configuration
    parser.add_argument(
        "--vector-size",
        type=int,
        default=300,
        help="Size of word vectors (default: 300)"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Context window size (default: 5)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum word frequency (default: 5)"
    )
    
    # Pattern configuration
    parser.add_argument(
        "--use-prime-patterns",
        action="store_true",
        default=True,
        help="Use prime-number skip patterns (default: True)"
    )
    
    parser.add_argument(
        "--use-log-patterns",
        action="store_true", 
        default=True,
        help="Use logarithmic skip patterns (default: True)"
    )
    
    parser.add_argument(
        "--max-prime-distance",
        type=int,
        default=50,
        help="Maximum distance for prime patterns (default: 50)"
    )
    
    # Preprocessing configuration
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Don't remove stopwords"
    )
    
    parser.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Don't lemmatize tokens"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AI Text Vectorization Agent Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load input texts
        input_path = Path(args.input)
        if input_path.is_file():
            logger.info(f"Loading single file: {input_path}")
            texts = [load_text_file(str(input_path))]
        elif input_path.is_dir():
            logger.info(f"Loading texts from directory: {input_path}")
            texts = load_texts_from_directory(str(input_path))
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        if not texts:
            raise ValueError("No texts loaded for training")
        
        logger.info(f"Loaded {len(texts)} text documents")
        
        # Configure preprocessing
        preprocessing_config = PreprocessingConfig(
            remove_stopwords=not args.no_stopwords,
            lemmatize=not args.no_lemmatize
        )
        
        # Configure vectorization
        vector_config = VectorConfig(
            vector_size=args.vector_size,
            window=args.window,
            epochs=args.epochs,
            min_count=args.min_count,
            use_prime_patterns=args.use_prime_patterns,
            use_log_patterns=args.use_log_patterns,
            max_prime_distance=args.max_prime_distance
        )
        
        # Initialize agent
        logger.info("Initializing Vectorization Agent")
        agent = VectorizationAgent(config=vector_config)
        
        # Start training
        start_time = time.time()
        logger.info("Starting model training...")
        
        model = agent.train(texts, model_path=args.model)
        
        training_time = time.time() - start_time
        
        # Log training results
        model_info = agent.get_model_info()
        logger.info("Training completed successfully!")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Model info: {model_info}")
        
        # Test the model with a few similarity queries
        logger.info("Testing trained model...")
        test_words = ["good", "bad", "love", "hate", "big", "small"]
        
        for word in test_words:
            if word in agent.model.wv:
                similar = agent.find_similar(word, topn=3)
                logger.info(f"Similar to '{word}': {similar}")
        
        logger.info(f"Model saved to: {args.model}")
        logger.info("Training process completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
