"""
Correlation Agent for semantic relationship analysis.

Performs statistical correlation analysis on vector spaces and
identifies semantic clustering patterns.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    similarity_threshold: float = 0.7
    cluster_count: int = 10
    min_cluster_size: int = 3
    max_iterations: int = 100


class CorrelationAgent:
    """
    Correlation analysis agent for semantic vector relationships.
    
    Features:
    - Statistical correlation analysis
    - Semantic clustering identification
    - Pattern mining and discovery
    - Association strength metrics
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """Initialize the correlation agent."""
        self.config = config or CorrelationConfig()
        self.correlation_matrix: Optional[np.ndarray] = None
        self.clusters: Optional[Dict[int, List[str]]] = None
        self.cluster_centers: Optional[np.ndarray] = None
        
        logger.info("CorrelationAgent initialized")
    
    def analyze_correlations(self, word_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze correlations between word vectors.
        
        Args:
            word_vectors: Dictionary mapping words to their vector representations
            
        Returns:
            Dictionary with correlation analysis results
        """
        if not word_vectors:
            return {}
        
        words = list(word_vectors.keys())
        vectors = np.array(list(word_vectors.values()))
        
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(vectors)
        
        # Find highly correlated pairs
        high_correlations = self._find_high_correlations(words, self.correlation_matrix)
        
        # Calculate overall statistics
        mean_correlation = np.mean(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)])
        std_correlation = np.std(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)])
        
        return {
            "mean_correlation": float(mean_correlation),
            "std_correlation": float(std_correlation),
            "high_correlation_pairs": high_correlations,
            "vocabulary_size": len(words)
        }
    
    def perform_clustering(self, word_vectors: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Perform semantic clustering on word vectors.
        
        Args:
            word_vectors: Dictionary mapping words to vectors
            
        Returns:
            Dictionary mapping cluster IDs to word lists
        """
        if not word_vectors:
            return {}
        
        words = list(word_vectors.keys())
        vectors = np.array(list(word_vectors.values()))
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=min(self.config.cluster_count, len(words)),
            max_iter=self.config.max_iterations,
            random_state=42
        )
        
        cluster_labels = kmeans.fit_predict(vectors)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Group words by cluster
        clusters = {}
        for word, label in zip(words, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(word)
        
        # Filter clusters by minimum size
        filtered_clusters = {
            cid: words_list for cid, words_list in clusters.items()
            if len(words_list) >= self.config.min_cluster_size
        }
        
        self.clusters = filtered_clusters
        logger.info(f"Created {len(filtered_clusters)} semantic clusters")
        
        return filtered_clusters
    
    def find_semantic_groups(self, word_vectors: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Find semantic groups using similarity analysis.
        
        Args:
            word_vectors: Dictionary mapping words to vectors
            
        Returns:
            List of semantic groups with metadata
        """
        if not word_vectors:
            return []
        
        words = list(word_vectors.keys())
        vectors = np.array(list(word_vectors.values()))
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Find semantic groups
        groups = []
        visited = set()
        
        for i, word in enumerate(words):
            if word in visited:
                continue
            
            # Find similar words
            similar_indices = np.where(similarity_matrix[i] > self.config.similarity_threshold)[0]
            similar_words = [words[j] for j in similar_indices if words[j] not in visited]
            
            if len(similar_words) >= self.config.min_cluster_size:
                # Calculate group statistics
                group_vectors = vectors[similar_indices]
                centroid = np.mean(group_vectors, axis=0)
                cohesion = np.mean([similarity_matrix[i][j] for j in similar_indices])
                
                groups.append({
                    "words": similar_words,
                    "size": len(similar_words),
                    "centroid": centroid.tolist(),
                    "cohesion": float(cohesion),
                    "representative": word
                })
                
                visited.update(similar_words)
        
        logger.info(f"Found {len(groups)} semantic groups")
        return groups
    
    def calculate_association_strength(self, word1: str, word2: str, word_vectors: Dict[str, np.ndarray]) -> Optional[float]:
        """Calculate association strength between two words."""
        if word1 not in word_vectors or word2 not in word_vectors:
            return None
        
        vector1 = word_vectors[word1]
        vector2 = word_vectors[word2]
        
        # Cosine similarity as association strength
        similarity = cosine_similarity([vector1], [vector2])[0][0]
        return float(similarity)
    
    def _find_high_correlations(self, words: List[str], correlation_matrix: np.ndarray) -> List[Tuple[str, str, float]]:
        """Find word pairs with high correlation."""
        high_correlations = []
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                correlation = correlation_matrix[i][j]
                if abs(correlation) > self.config.similarity_threshold:
                    high_correlations.append((words[i], words[j], float(correlation)))
        
        # Sort by correlation strength
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_correlations
    
    def get_cluster_summary(self) -> Dict:
        """Get summary of clustering results."""
        if not self.clusters:
            return {"clusters": 0, "total_words": 0}
        
        cluster_sizes = [len(words) for words in self.clusters.values()]
        
        return {
            "clusters": len(self.clusters),
            "total_words": sum(cluster_sizes),
            "avg_cluster_size": np.mean(cluster_sizes),
            "largest_cluster": max(cluster_sizes),
            "smallest_cluster": min(cluster_sizes)
        }
