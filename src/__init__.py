"""
AI-Driven Text Vectorization Agent

Core package for advanced semantic analysis through multidimensional vector field generation.
Implements novel skip-gram methodologies with prime-number and logarithmic interval algorithms.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "team@vectorization-ai.com"

from .agents.vectorization import VectorizationAgent
from .agents.correlation import CorrelationAgent
from .agents.orchestration import AnalysisOrchestrator

__all__ = [
    "VectorizationAgent",
    "CorrelationAgent", 
    "AnalysisOrchestrator"
]
