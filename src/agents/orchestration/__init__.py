"""
Analysis Orchestrator for coordinating multi-agent workflows.

Manages communication between vectorization and correlation agents,
coordinates complex analysis workflows, and maintains system state.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from ..vectorization import VectorizationAgent, VectorConfig
from ..correlation import CorrelationAgent, CorrelationConfig

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    timestamp: float = field(default_factory=time.time)
    vectorization_info: Dict = field(default_factory=dict)
    correlation_analysis: Dict = field(default_factory=dict)
    semantic_clusters: Dict = field(default_factory=dict)
    processing_time: float = 0.0
    text_count: int = 0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class OrchestrationConfig:
    """Configuration for the analysis orchestrator."""
    save_intermediate_results: bool = True
    results_directory: str = "results"
    enable_parallel_processing: bool = True
    max_concurrent_tasks: int = 4


class AnalysisOrchestrator:
    """
    Orchestrates complex analysis workflows across multiple agents.
    
    Features:
    - Multi-agent coordination
    - Workflow management
    - Result aggregation
    - State persistence
    - Error handling and recovery
    """
    
    def __init__(self, 
                 vector_config: Optional[VectorConfig] = None,
                 correlation_config: Optional[CorrelationConfig] = None,
                 orchestration_config: Optional[OrchestrationConfig] = None):
        """Initialize the analysis orchestrator."""
        self.vector_config = vector_config or VectorConfig()
        self.correlation_config = correlation_config or CorrelationConfig()
        self.orchestration_config = orchestration_config or OrchestrationConfig()
        
        # Initialize agents
        self.vectorization_agent = VectorizationAgent(self.vector_config)
        self.correlation_agent = CorrelationAgent(self.correlation_config)
        
        # State management
        self.current_analysis: Optional[AnalysisResult] = None
        self.analysis_history: List[AnalysisResult] = []
        
        # Setup results directory
        self.results_path = Path(self.orchestration_config.results_directory)
        self.results_path.mkdir(exist_ok=True)
        
        logger.info("AnalysisOrchestrator initialized")
    
    async def run_full_analysis(self, texts: List[str], save_model_path: Optional[str] = None) -> AnalysisResult:
        """
        Run complete analysis pipeline on input texts.
        
        Args:
            texts: List of input text documents
            save_model_path: Optional path to save the trained model
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        result = AnalysisResult(text_count=len(texts))
        
        try:
            logger.info(f"Starting full analysis on {len(texts)} texts")
            
            # Phase 1: Vectorization
            logger.info("Phase 1: Training vectorization model")
            model = await self._run_vectorization(texts, save_model_path)
            result.vectorization_info = self.vectorization_agent.get_model_info()
            
            # Phase 2: Extract word vectors
            logger.info("Phase 2: Extracting word vectors")
            word_vectors = self._extract_word_vectors()
            
            # Phase 3: Correlation analysis
            logger.info("Phase 3: Running correlation analysis")
            correlation_results = await self._run_correlation_analysis(word_vectors)
            result.correlation_analysis = correlation_results
            
            # Phase 4: Semantic clustering
            logger.info("Phase 4: Performing semantic clustering")
            clustering_results = await self._run_clustering_analysis(word_vectors)
            result.semantic_clusters = clustering_results
            
            # Calculate processing time
            result.processing_time = time.time() - start_time
            result.success = True
            
            logger.info(f"Full analysis completed in {result.processing_time:.2f} seconds")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {e}")
        
        # Save results
        if self.orchestration_config.save_intermediate_results:
            self._save_analysis_result(result)
        
        self.current_analysis = result
        self.analysis_history.append(result)
        
        return result
    
    async def _run_vectorization(self, texts: List[str], save_model_path: Optional[str] = None):
        """Run vectorization in async context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.vectorization_agent.train,
            texts,
            save_model_path
        )
    
    def _extract_word_vectors(self) -> Dict[str, Any]:
        """Extract word vectors from trained model."""
        if not self.vectorization_agent.model or not self.vectorization_agent.vocabulary:
            return {}
        
        word_vectors = {}
        for word in list(self.vectorization_agent.vocabulary.keys())[:1000]:  # Limit for performance
            vector = self.vectorization_agent.get_vector(word)
            if vector is not None:
                word_vectors[word] = vector
        
        logger.info(f"Extracted {len(word_vectors)} word vectors")
        return word_vectors
    
    async def _run_correlation_analysis(self, word_vectors: Dict[str, Any]) -> Dict:
        """Run correlation analysis in async context."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.correlation_agent.analyze_correlations,
            word_vectors
        )
    
    async def _run_clustering_analysis(self, word_vectors: Dict[str, Any]) -> Dict:
        """Run clustering analysis in async context."""
        if not word_vectors:
            return {}
        
        loop = asyncio.get_event_loop()
        
        # Run clustering and semantic grouping in parallel
        clustering_task = loop.run_in_executor(
            None,
            self.correlation_agent.perform_clustering,
            word_vectors
        )
        
        semantic_groups_task = loop.run_in_executor(
            None,
            self.correlation_agent.find_semantic_groups,
            word_vectors
        )
        
        clusters, semantic_groups = await asyncio.gather(clustering_task, semantic_groups_task)
        
        return {
            "clusters": clusters,
            "semantic_groups": semantic_groups,
            "cluster_summary": self.correlation_agent.get_cluster_summary()
        }
    
    def _save_analysis_result(self, result: AnalysisResult) -> None:
        """Save analysis result to disk."""
        timestamp = int(result.timestamp)
        filename = f"analysis_result_{timestamp}.json"
        filepath = self.results_path / filename
        
        # Convert result to JSON-serializable format
        result_dict = {
            "timestamp": result.timestamp,
            "vectorization_info": result.vectorization_info,
            "correlation_analysis": result.correlation_analysis,
            "semantic_clusters": {
                k: v for k, v in result.semantic_clusters.items()
                if k != "clusters" or isinstance(v, (dict, list, str, int, float))
            },
            "processing_time": result.processing_time,
            "text_count": result.text_count,
            "success": result.success,
            "error_message": result.error_message
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis result saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save analysis result: {e}")
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of all analysis runs."""
        if not self.analysis_history:
            return {"total_analyses": 0}
        
        successful_analyses = [r for r in self.analysis_history if r.success]
        failed_analyses = [r for r in self.analysis_history if not r.success]
        
        if successful_analyses:
            avg_processing_time = sum(r.processing_time for r in successful_analyses) / len(successful_analyses)
            total_texts_processed = sum(r.text_count for r in successful_analyses)
        else:
            avg_processing_time = 0
            total_texts_processed = 0
        
        return {
            "total_analyses": len(self.analysis_history),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(failed_analyses),
            "avg_processing_time": avg_processing_time,
            "total_texts_processed": total_texts_processed,
            "last_analysis_time": self.analysis_history[-1].timestamp if self.analysis_history else None
        }
    
    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained vectorization model."""
        try:
            self.vectorization_agent.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    async def query_similar_words(self, word: str, topn: int = 10) -> List:
        """Query similar words using the trained model."""
        if not self.vectorization_agent.model:
            return []
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.vectorization_agent.find_similar,
            word,
            topn
        )
    
    async def calculate_word_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Calculate similarity between two words."""
        if not self.vectorization_agent.model:
            return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.vectorization_agent.calculate_similarity,
            word1,
            word2
        )
