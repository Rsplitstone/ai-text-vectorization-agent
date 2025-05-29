"""
Novel Skip-Pattern Algorithms

Implements prime-number and logarithmic interval-based context extraction
for enhanced semantic vector training.
"""

import numpy as np
from typing import List, Tuple, Iterator
import math
from abc import ABC, abstractmethod


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate prime numbers up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]


class SkipPattern(ABC):
    """Abstract base class for skip pattern implementations."""
    
    @abstractmethod
    def extract_sequences(self, tokens: List[str]) -> List[List[str]]:
        """Extract sequences based on the pattern algorithm."""
        pass


class PrimeSkipPattern(SkipPattern):
    """
    Prime-number based context extraction pattern.
    
    Uses prime number intervals to create non-linear context windows
    that capture distant semantic relationships.
    """
    
    def __init__(self, max_distance: int = 50):
        """Initialize with maximum distance for prime generation."""
        self.max_distance = max_distance
        self.primes = sieve_of_eratosthenes(max_distance)
    
    def extract_sequences(self, tokens: List[str]) -> List[List[str]]:
        """
        Extract sequences using prime-number skip patterns.
        
        For each target word, create context windows at prime-number distances
        to capture long-range semantic dependencies.
        """
        sequences = []
        
        for i, target in enumerate(tokens):
            prime_sequence = [target]
            
            # Extract context at prime distances
            for prime in self.primes:
                # Forward context
                if i + prime < len(tokens):
                    prime_sequence.append(tokens[i + prime])
                
                # Backward context  
                if i - prime >= 0:
                    prime_sequence.append(tokens[i - prime])
            
            # Only add sequences with sufficient context
            if len(prime_sequence) > 3:
                sequences.append(prime_sequence)
        
        return sequences
    
    def extract_context_pairs(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Extract (target, context) pairs using prime skip patterns."""
        pairs = []
        
        for i, target in enumerate(tokens):
            for prime in self.primes:
                # Forward prime context
                if i + prime < len(tokens):
                    pairs.append((target, tokens[i + prime]))
                
                # Backward prime context
                if i - prime >= 0:
                    pairs.append((target, tokens[i - prime]))
        
        return pairs


class LogarithmicSkipPattern(SkipPattern):
    """
    Logarithmic weighting scheme for context extraction.
    
    Applies natural log-based distance weighting to emphasize
    nearby context while maintaining distant relationships.
    """
    
    def __init__(self, max_distance: int = 20, base_weight: float = 1.0):
        """Initialize with maximum distance and base weighting."""
        self.max_distance = max_distance
        self.base_weight = base_weight
    
    def extract_sequences(self, tokens: List[str]) -> List[List[str]]:
        """
        Extract sequences with logarithmic distance weighting.
        
        Creates weighted context windows where closer words have
        higher probability of inclusion based on log weighting.
        """
        sequences = []
        
        for i, target in enumerate(tokens):
            log_sequence = [target]
            
            # Generate context with logarithmic sampling
            for distance in range(1, min(self.max_distance + 1, len(tokens))):
                weight = self._log_weight(distance)
                
                # Probabilistic inclusion based on log weight
                if np.random.random() < weight:
                    # Forward context
                    if i + distance < len(tokens):
                        log_sequence.append(tokens[i + distance])
                    
                    # Backward context
                    if i - distance >= 0:
                        log_sequence.append(tokens[i - distance])
            
            if len(log_sequence) > 2:
                sequences.append(log_sequence)
        
        return sequences
    
    def _log_weight(self, distance: int) -> float:
        """Calculate logarithmic weight for given distance."""
        return self.base_weight / (1.0 + np.log(distance + 1))
    
    def get_weighted_context(self, tokens: List[str], center_idx: int) -> List[Tuple[str, float]]:
        """Get context words with their logarithmic weights."""
        weighted_context = []
        
        for i, token in enumerate(tokens):
            if i != center_idx:
                distance = abs(i - center_idx)
                if distance <= self.max_distance:
                    weight = self._log_weight(distance)
                    weighted_context.append((token, weight))
        
        return weighted_context


class CombinedSkipPattern(SkipPattern):
    """
    Combined pattern using both prime and logarithmic methods.
    
    Integrates prime-number context extraction with logarithmic
    weighting for comprehensive semantic relationship capture.
    """
    
    def __init__(self, max_prime_distance: int = 50, max_log_distance: int = 20):
        """Initialize with separate distance limits for each pattern."""
        self.prime_pattern = PrimeSkipPattern(max_prime_distance)
        self.log_pattern = LogarithmicSkipPattern(max_log_distance)
    
    def extract_sequences(self, tokens: List[str]) -> List[List[str]]:
        """Extract sequences using combined prime and logarithmic patterns."""
        prime_sequences = self.prime_pattern.extract_sequences(tokens)
        log_sequences = self.log_pattern.extract_sequences(tokens)
        
        # Combine and return unique sequences
        all_sequences = prime_sequences + log_sequences
        
        # Remove duplicates while preserving order
        unique_sequences = []
        seen = set()
        
        for seq in all_sequences:
            seq_tuple = tuple(seq)
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique_sequences.append(seq)
        
        return unique_sequences
    
    def extract_enhanced_training_data(self, tokens: List[str]) -> Dict[str, List[List[str]]]:
        """
        Extract comprehensive training data with pattern attribution.
        
        Returns:
            Dictionary with 'prime', 'logarithmic', and 'combined' sequences
        """
        return {
            'prime': self.prime_pattern.extract_sequences(tokens),
            'logarithmic': self.log_pattern.extract_sequences(tokens),
            'combined': self.extract_sequences(tokens)
        }
