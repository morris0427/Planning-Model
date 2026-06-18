"""
Abstract base class for data generation.

Both Blocks World and 8-Puzzle inherit from this to ensure
consistent interfaces and enable domain-agnostic training.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import json
import numpy as np


class PlanningDataset(ABC):
    """Abstract base class for planning datasets."""
    
    def __init__(
        self,
        difficulty_range: Tuple[int, int],
        num_samples: int,
        use_world_model: bool = False,
        seed: int = 42
    ):
        """
        Initialize dataset.
        
        Args:
            difficulty_range: (min, max) difficulty (e.g., number of moves)
            num_samples: Number of problems to generate
            use_world_model: If True, include intermediate states
            seed: Random seed
        """
        self.difficulty_range = difficulty_range
        self.num_samples = num_samples
        self.use_world_model = use_world_model
        self.seed = seed
        
        np.random.seed(seed)
        
        self.problems = []
        self.vocab = {}
        self.inv_vocab = {}
        
    @abstractmethod
    def generate_problem(self, difficulty: int) -> Dict[str, Any]:
        """
        Generate a single problem.
        
        Args:
            difficulty: Problem difficulty (domain-specific interpretation)
            
        Returns:
            Dictionary with keys:
                - start_state: Initial state
                - goal_state: Target state
                - solution_moves: List of actions
                - solution_states: List of intermediate states (if WM)
                - num_moves: Number of moves in solution
        """
        pass
    
    @abstractmethod
    def encode_sequence(self, problem: Dict[str, Any]) -> List[int]:
        """
        Encode problem as token sequence.
        
        For baseline: [START, start, goal, action1, action2, ..., END]
        For WM: [START, start, goal, action1, state1, action2, state2, ..., END]
        
        Args:
            problem: Problem dictionary from generate_problem()
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode_sequence(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        Decode token sequence back to problem structure.
        
        Args:
            token_ids: Encoded sequence
            
        Returns:
            Dictionary with decoded states and actions
        """
        pass
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """
        Generate complete dataset.
        
        Returns:
            List of encoded problems
        """
        self.problems = []
        
        min_diff, max_diff = self.difficulty_range
        
        for _ in range(self.num_samples):
            # Sample difficulty uniformly
            difficulty = np.random.randint(min_diff, max_diff + 1)
            
            # Generate problem
            problem = self.generate_problem(difficulty)
            
            # Encode to sequence
            sequence = self.encode_sequence(problem)
            
            # Store
            self.problems.append({
                'sequence': sequence,
                'length': len(sequence),
                'num_moves': problem['num_moves'],
                'problem_idx': len(self.problems)
            })
        
        return self.problems
    
    def save_dataset(self, filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.problems, f, indent=2)
    
    @classmethod
    def load_dataset(cls, filepath: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length in dataset."""
        if not self.problems:
            # Estimate based on difficulty range
            max_diff = self.difficulty_range[1]
            if self.use_world_model:
                # START + 2 states + (action + state) * n + END
                return 1 + 2 * self._estimate_state_tokens() + \
                       (2 + self._estimate_state_tokens()) * max_diff + 1
            else:
                # START + 2 states + n actions + END
                return 1 + 2 * self._estimate_state_tokens() + 2 * max_diff + 1
        else:
            return max(p['length'] for p in self.problems)
    
    @abstractmethod
    def _estimate_state_tokens(self) -> int:
        """Estimate number of tokens per state (domain-specific)."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.problems:
            return {}
        
        lengths = [p['length'] for p in self.problems]
        moves = [p['num_moves'] for p in self.problems]
        
        return {
            'num_problems': len(self.problems),
            'vocab_size': self.get_vocab_size(),
            'max_seq_length': max(lengths),
            'avg_seq_length': np.mean(lengths),
            'difficulty_range': self.difficulty_range,
            'avg_num_moves': np.mean(moves),
            'min_num_moves': min(moves),
            'max_num_moves': max(moves),
            'use_world_model': self.use_world_model
        }
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        print("Dataset Statistics:")
        print(f"  Problems: {stats['num_problems']}")
        print(f"  Vocabulary size: {stats['vocab_size']}")
        print(f"  Sequence length: {stats['avg_seq_length']:.1f} avg, {stats['max_seq_length']} max")
        print(f"  Difficulty: {stats['avg_num_moves']:.1f} avg, {stats['min_num_moves']}-{stats['max_num_moves']} range")
        print(f"  World model: {stats['use_world_model']}")
    
    def print_example(self, idx: int = 0):
        """Print a decoded example."""
        if not self.problems:
            print("No problems generated yet")
            return
        
        problem = self.problems[idx]
        decoded = self.decode_sequence(problem['sequence'])
        
        print(f"\nExample problem #{idx}:")
        print(f"  Difficulty: {problem['num_moves']} moves")
        print(f"  Sequence length: {problem['length']} tokens")
        print(f"  Start state: {decoded.get('start_state', 'N/A')}")
        print(f"  Goal state: {decoded.get('goal_state', 'N/A')}")
        print(f"  Solution: {decoded.get('actions', 'N/A')}")


class DatasetFactory:
    """Factory for creating datasets."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register dataset classes."""
        def decorator(dataset_class):
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def create(
        cls,
        domain: str,
        difficulty_range: Tuple[int, int],
        num_samples: int,
        use_world_model: bool = False,
        seed: int = 42
    ) -> PlanningDataset:
        """
        Create dataset for specified domain.
        
        Args:
            domain: "blocks_world" or "8_puzzle"
            difficulty_range: Problem difficulty range
            num_samples: Number of samples
            use_world_model: Include intermediate states
            seed: Random seed
            
        Returns:
            Dataset instance
        """
        # Auto-import domain module if not already registered
        if domain not in cls._registry:
            try:
                if domain == "blocks_world":
                    from . import blocks_world
                elif domain == "8_puzzle":
                    from . import puzzle
                else:
                    # Try generic import
                    __import__(f'data.{domain}')
            except ImportError as e:
                print(f"⚠️  Could not import domain module: {domain}")
                print(f"   Error: {e}")
                print(f"   Make sure data/{domain}.py exists")
        
        if domain not in cls._registry:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls._registry.keys())}")
        
        dataset_class = cls._registry[domain]
        return dataset_class(
            difficulty_range=difficulty_range,
            num_samples=num_samples,
            use_world_model=use_world_model,
            seed=seed
        )
