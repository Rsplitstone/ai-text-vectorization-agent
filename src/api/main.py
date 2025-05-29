"""
FastAPI Application for Text Vectorization Agent

RESTful API service providing vector generation, similarity search,
and semantic analysis capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Union
import uvicorn
import logging
import numpy as np
from pathlib import Path
import asyncio
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.vectorization import VectorizationAgent, VectorConfig
from utilities.data_processing.preprocessing import TextPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Text Vectorization Agent API",
    description="Advanced semantic analysis through multidimensional vector field generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
vectorization_agent: Optional[VectorizationAgent] = None
preprocessor = TextPreprocessor()

# Pydantic models for API
class VectorRequest(BaseModel):
    """Request model for getting word vectors."""
    word: str = Field(..., description="Word to get vector for")

class SimilarityRequest(BaseModel):
    """Request model for word similarity."""
    word1: str = Field(..., description="First word")
    word2: str = Field(..., description="Second word")

class SimilarWordsRequest(BaseModel):
    """Request model for finding similar words."""
    word: str = Field(..., description="Target word")
    topn: int = Field(10, description="Number of similar words to return", ge=1, le=100)

class TrainingRequest(BaseModel):
    """Request model for training the model."""
    texts: List[str] = Field(..., description="List of training texts")
    config: Optional[Dict] = Field(None, description="Training configuration")

class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze")

class VectorResponse(BaseModel):
    """Response model for vector data."""
    word: str
    vector: Optional[List[float]]
    available: bool

class SimilarityResponse(BaseModel):
    """Response model for similarity scores."""
    word1: str
    word2: str
    similarity: Optional[float]
    available: bool

class SimilarWordsResponse(BaseModel):
    """Response model for similar words."""
    word: str
    similar_words: List[Tuple[str, float]]
    available: bool

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    loaded: bool
    vocabulary_size: int
    vector_size: int
    total_words: Optional[int] = None
    epochs: Optional[int] = None

class StatusResponse(BaseModel):
    """Response model for API status."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting AI Text Vectorization Agent API")
    
    # Try to load a pre-trained model if available
    model_path = "artifacts/vectors.model"
    if Path(model_path).exists():
        global vectorization_agent
        vectorization_agent = VectorizationAgent()
        try:
            vectorization_agent.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            vectorization_agent = None


@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status."""
    return StatusResponse(
        status="running",
        model_loaded=vectorization_agent is not None,
        version="1.0.0"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-05-29"}


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if not vectorization_agent:
        return ModelInfoResponse(
            loaded=False,
            vocabulary_size=0,
            vector_size=0
        )
    
    info = vectorization_agent.get_model_info()
    return ModelInfoResponse(
        loaded=True,
        vocabulary_size=info.get("vocabulary_size", 0),
        vector_size=info.get("vector_size", 0),
        total_words=info.get("total_words"),
        epochs=info.get("epochs")
    )


@app.post("/vector", response_model=VectorResponse)
async def get_word_vector(request: VectorRequest):
    """Get vector representation for a word."""
    if not vectorization_agent:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    vector = vectorization_agent.get_vector(request.word)
    
    return VectorResponse(
        word=request.word,
        vector=vector.tolist() if vector is not None else None,
        available=vector is not None
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """Calculate similarity between two words."""
    if not vectorization_agent:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    similarity = vectorization_agent.calculate_similarity(request.word1, request.word2)
    
    return SimilarityResponse(
        word1=request.word1,
        word2=request.word2,
        similarity=similarity,
        available=similarity is not None
    )


@app.post("/similar", response_model=SimilarWordsResponse)
async def find_similar_words(request: SimilarWordsRequest):
    """Find most similar words to the given word."""
    if not vectorization_agent:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    similar_words = vectorization_agent.find_similar(request.word, request.topn)
    
    return SimilarWordsResponse(
        word=request.word,
        similar_words=similar_words,
        available=len(similar_words) > 0
    )


@app.post("/analyze")
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text and return semantic insights."""
    if not vectorization_agent:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Preprocess the text
    tokens = preprocessor.process(request.text)
    
    # Get vectors for available words
    word_vectors = {}
    available_words = []
    
    for token in tokens:
        vector = vectorization_agent.get_vector(token)
        if vector is not None:
            word_vectors[token] = vector.tolist()
            available_words.append(token)
    
    # Calculate text-level statistics
    if available_words:
        vectors_array = np.array([vectorization_agent.get_vector(word) for word in available_words])
        mean_vector = np.mean(vectors_array, axis=0)
        vector_variance = np.var(vectors_array, axis=0)
        
        analysis = {
            "total_tokens": len(tokens),
            "available_words": len(available_words),
            "coverage": len(available_words) / len(tokens) if tokens else 0,
            "word_vectors": word_vectors,
            "mean_vector": mean_vector.tolist(),
            "vector_variance": vector_variance.tolist(),
            "semantic_density": float(np.mean(vector_variance))
        }
    else:
        analysis = {
            "total_tokens": len(tokens),
            "available_words": 0,
            "coverage": 0,
            "word_vectors": {},
            "mean_vector": None,
            "vector_variance": None,
            "semantic_density": 0
        }
    
    return analysis


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the vectorization model on provided texts."""
    try:
        # Parse configuration
        config_dict = request.config or {}
        config = VectorConfig(**config_dict)
        
        # Initialize new agent
        global vectorization_agent
        vectorization_agent = VectorizationAgent(config=config)
        
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            request.texts,
            "artifacts/vectors.model"
        )
        
        return {
            "message": "Training started",
            "status": "training",
            "text_count": len(request.texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")


async def train_model_background(texts: List[str], model_path: str):
    """Background task for model training."""
    try:
        logger.info(f"Starting background training on {len(texts)} texts")
        vectorization_agent.train(texts, model_path=model_path)
        logger.info("Background training completed successfully")
    except Exception as e:
        logger.error(f"Background training failed: {e}")


@app.get("/vocabulary")
async def get_vocabulary():
    """Get the model's vocabulary."""
    if not vectorization_agent or not vectorization_agent.vocabulary:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "vocabulary_size": len(vectorization_agent.vocabulary),
        "words": list(vectorization_agent.vocabulary.keys())[:1000]  # Limit to first 1000
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
