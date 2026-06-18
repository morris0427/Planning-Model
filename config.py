"""
Shared configuration system for Blocks World and 8-Puzzle experiments.

This module defines all experimental configurations in a factored way,
allowing easy ablations and sweeps across both domains.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
from enum import Enum


class SplitType(Enum):
    """Types of train/test splits for compositional generalization."""
    IN_DISTRIBUTION = "in_distribution"      # Train 1-6, Test 1-6
    PRODUCTIVITY = "productivity"             # Train 1-4, Test 5-8
    SYSTEMATICITY = "systematicity"           # Train simple, Test complex
    LENGTH_EXTRAPOLATION = "length_extrap"    # Train odd, Test even
    FIXED_TO_VARIED = "fixed_to_varied"       # Train fixed length, Test varied


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model size
    d_model: int = 64           # Embedding dimension
    n_heads: int = 4            # Number of attention heads
    d_ff: int = 256            # Feedforward dimension
    n_layers: int = 2          # Number of transformer layers
    
    # Architecture features
    weight_sharing: bool = False  # Share weights across layers
    dropout: float = 0.1
    
    # Training type
    use_world_model: bool = False  # Predict intermediate states
    
    # Optimization
    learning_rate: float = 0.0001  # CRITICAL: 0.0001 not 0.001 (prevents divergence)
    batch_size: int = 32
    max_epochs: int = 100
    
    # Model name (auto-generated if None)
    name: Optional[str] = None
    
    def __post_init__(self):
        """Generate descriptive name if not provided."""
        if self.name is None:
            size = "tiny" if self.d_model <= 32 else "small" if self.d_model <= 64 else "medium" if self.d_model <= 128 else "large"
            wm = "WM" if self.use_world_model else "Base"
            ws = "Shared" if self.weight_sharing else ""
            self.name = f"{size}_{wm}{ws}"
    
    @property
    def param_count_estimate(self) -> int:
        """Rough estimate of parameter count."""
        # Embeddings + layers + output
        vocab_size = 100  # approximate
        embeddings = vocab_size * self.d_model
        
        if self.weight_sharing:
            # Only count one layer, applied n times
            layer_params = (
                4 * self.d_model * self.d_model +  # Attention
                2 * self.d_model * self.d_ff        # FFN
            )
            layers = layer_params
        else:
            layer_params = (
                4 * self.d_model * self.d_model +
                2 * self.d_model * self.d_ff
            )
            layers = layer_params * self.n_layers
        
        output_layer = vocab_size * self.d_model
        
        return embeddings + layers + output_layer


@dataclass
class DataConfig:
    """Data generation and splitting configuration."""
    
    # Domain
    domain: Literal["blocks_world", "8_puzzle"] = "blocks_world"
    
    # Split type
    split_type: SplitType = SplitType.IN_DISTRIBUTION
    
    # Problem generation
    num_train_samples: int = 20000
    num_test_samples: int = 2000
    
    # Difficulty range (domain-specific interpretation)
    train_difficulty_range: Tuple[int, int] = (1, 6)
    test_difficulty_range: Tuple[int, int] = (1, 6)
    
    # For fixed-length training
    fixed_train_length: Optional[int] = None
    
    # Data augmentation
    augment_data: bool = False
    
    def get_split_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get train and test ranges based on split type."""
        
        if self.split_type == SplitType.IN_DISTRIBUTION:
            # Standard: same distribution
            return self.train_difficulty_range, self.test_difficulty_range
        
        elif self.split_type == SplitType.PRODUCTIVITY:
            # Train on easier, test on harder
            max_train = self.train_difficulty_range[1]
            return (
                self.train_difficulty_range,
                (max_train + 1, max_train + 4)
            )
        
        elif self.split_type == SplitType.SYSTEMATICITY:
            # Both use same range, but different patterns
            # (pattern selection is domain-specific)
            return self.train_difficulty_range, self.test_difficulty_range
        
        elif self.split_type == SplitType.LENGTH_EXTRAPOLATION:
            # Odd vs even lengths
            min_len, max_len = self.train_difficulty_range
            train_lengths = list(range(min_len, max_len + 1, 2))  # odd
            test_lengths = list(range(min_len + 1, max_len + 1, 2))  # even
            return (
                (min(train_lengths), max(train_lengths)),
                (min(test_lengths), max(test_lengths))
            )
        
        elif self.split_type == SplitType.FIXED_TO_VARIED:
            # Train on fixed length, test on varied
            fixed = self.fixed_train_length or self.train_difficulty_range[0]
            return (
                (fixed, fixed),
                self.test_difficulty_range
            )
        
        else:
            return self.train_difficulty_range, self.test_difficulty_range


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    model: ModelConfig
    data: DataConfig
    
    # Experiment metadata
    experiment_name: str = "unnamed_experiment"
    seed: int = 42
    save_dir: str = "./results"
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500
    save_checkpoint: bool = True
    
    def __str__(self) -> str:
        """Human-readable description."""
        return (
            f"Experiment: {self.experiment_name}\n"
            f"  Domain: {self.data.domain}\n"
            f"  Model: {self.model.name} ({self.model.param_count_estimate:,} params)\n"
            f"  Split: {self.data.split_type.value}\n"
            f"  Train range: {self.data.train_difficulty_range}\n"
            f"  Test range: {self.data.test_difficulty_range}\n"
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

class ModelPresets:
    """Predefined model configurations for easy reference."""
    
    @staticmethod
    def tiny(use_world_model: bool = False, weight_sharing: bool = False) -> ModelConfig:
        """Tiny model: ~60K parameters."""
        return ModelConfig(
            d_model=32, n_heads=2, d_ff=128, n_layers=2,
            use_world_model=use_world_model,
            weight_sharing=weight_sharing
        )
    
    @staticmethod
    def small(use_world_model: bool = False, weight_sharing: bool = False) -> ModelConfig:
        """Small model: ~240K parameters (current default)."""
        return ModelConfig(
            d_model=64, n_heads=4, d_ff=256, n_layers=2,
            use_world_model=use_world_model,
            weight_sharing=weight_sharing
        )
    
    @staticmethod
    def medium(use_world_model: bool = False, weight_sharing: bool = False) -> ModelConfig:
        """Medium model: ~900K parameters."""
        return ModelConfig(
            d_model=128, n_heads=8, d_ff=512, n_layers=4,
            use_world_model=use_world_model,
            weight_sharing=weight_sharing
        )
    
    @staticmethod
    def large(use_world_model: bool = False, weight_sharing: bool = False) -> ModelConfig:
        """Large model: ~2M parameters."""
        return ModelConfig(
            d_model=256, n_heads=8, d_ff=1024, n_layers=4,
            use_world_model=use_world_model,
            weight_sharing=weight_sharing
        )


class DataPresets:
    """Predefined data configurations."""
    
    @staticmethod
    def blocks_world_standard() -> DataConfig:
        """Standard Blocks World setup (1-6 moves, in-distribution)."""
        return DataConfig(
            domain="blocks_world",
            split_type=SplitType.IN_DISTRIBUTION,
            train_difficulty_range=(1, 6),
            test_difficulty_range=(1, 6),
            num_train_samples=20000,
            num_test_samples=2000
        )
    
    @staticmethod
    def blocks_world_productivity() -> DataConfig:
        """Blocks World productivity split (train 1-4, test 5-8)."""
        return DataConfig(
            domain="blocks_world",
            split_type=SplitType.PRODUCTIVITY,
            train_difficulty_range=(1, 4),
            test_difficulty_range=(5, 8),
            num_train_samples=20000,
            num_test_samples=2000
        )
    
    @staticmethod
    def eight_puzzle_standard() -> DataConfig:
        """Standard 8-Puzzle setup (10-15 moves, in-distribution).
        
        Note: Using (10, 15) range to match previous tiny/small/medium experiments.
        """
        return DataConfig(
            domain="eight_puzzle",
            split_type=SplitType.IN_DISTRIBUTION,
            train_difficulty_range=(10, 15),  # Match previous runs
            test_difficulty_range=(10, 15),   # Match previous runs
            num_train_samples=5000,           # Match previous runs
            num_test_samples=500              # Match previous runs
        )
    
    @staticmethod
    def eight_puzzle_narrow() -> DataConfig:
        """8-Puzzle with narrow difficulty range (10-15 moves, slow generation)."""
        return DataConfig(
            domain="eight_puzzle",
            split_type=SplitType.IN_DISTRIBUTION,
            train_difficulty_range=(10, 15),
            test_difficulty_range=(10, 15),
            num_train_samples=5000,
            num_test_samples=500
        )
    
    @staticmethod
    def eight_puzzle_fast() -> DataConfig:
        """Fast 8-Puzzle for testing (5-20 moves, very quick generation)."""
        return DataConfig(
            domain="eight_puzzle",
            split_type=SplitType.IN_DISTRIBUTION,
            train_difficulty_range=(5, 20),
            test_difficulty_range=(5, 20),
            num_train_samples=1000,
            num_test_samples=100
        )
    
    @staticmethod
    def eight_puzzle_productivity() -> DataConfig:
        """8-Puzzle productivity split (train 10-12, test 13-18)."""
        return DataConfig(
            domain="eight_puzzle",
            split_type=SplitType.PRODUCTIVITY,
            train_difficulty_range=(10, 12),
            test_difficulty_range=(13, 18),
            num_train_samples=5000,
            num_test_samples=500
        )



# =============================================================================
# EXPERIMENT SWEEPS
# =============================================================================

def create_model_size_sweep(
    domain: str,
    split_type: SplitType = SplitType.IN_DISTRIBUTION
) -> list[ExperimentConfig]:
    """
    Create a sweep over model sizes for both baseline and world model.
    
    Returns 8 experiments:
    - tiny baseline/WM
    - small baseline/WM
    - medium baseline/WM
    - large baseline/WM
    """
    experiments = []
    
    # Get appropriate data config
    if domain == "blocks_world":
        if split_type == SplitType.PRODUCTIVITY:
            data_config = DataPresets.blocks_world_productivity()
        else:
            data_config = DataPresets.blocks_world_standard()
    else:  # eight_puzzle
        if split_type == SplitType.PRODUCTIVITY:
            data_config = DataPresets.eight_puzzle_productivity()
        else:
            data_config = DataPresets.eight_puzzle_standard()
    
    # Override split type
    data_config.split_type = split_type
    
    # Create configs for each size and model type
    for size_name, size_fn in [
        ("tiny", ModelPresets.tiny),
        ("small", ModelPresets.small),
        ("medium", ModelPresets.medium),
        ("large", ModelPresets.large),
    ]:
        for use_wm in [False, True]:
            model_config = size_fn(use_world_model=use_wm)
            
            experiment = ExperimentConfig(
                model=model_config,
                data=data_config,
                experiment_name=f"{domain}_{split_type.value}_{size_name}_{'wm' if use_wm else 'base'}"
            )
            experiments.append(experiment)
    
    return experiments


def create_weight_sharing_sweep(
    domain: str,
    model_size: str = "small"
) -> list[ExperimentConfig]:
    """
    Create a sweep over weight sharing for baseline and WM.
    
    Returns 4 experiments:
    - baseline no sharing
    - baseline with sharing
    - WM no sharing
    - WM with sharing
    """
    experiments = []
    
    # Get model preset function
    size_fn = getattr(ModelPresets, model_size)
    
    # Get data config
    data_config = (
        DataPresets.blocks_world_standard() if domain == "blocks_world"
        else DataPresets.eight_puzzle_standard()
    )
    
    for use_wm in [False, True]:
        for weight_sharing in [False, True]:
            model_config = size_fn(
                use_world_model=use_wm,
                weight_sharing=weight_sharing
            )
            
            experiment = ExperimentConfig(
                model=model_config,
                data=data_config,
                experiment_name=f"{domain}_{model_size}_{'wm' if use_wm else 'base'}_{'shared' if weight_sharing else 'standard'}"
            )
            experiments.append(experiment)
    
    return experiments


def create_full_ablation(domain: str) -> list[ExperimentConfig]:
    """
    Create complete ablation study:
    - 4 sizes × 2 model types × 2 weight sharing = 16 configs
    """
    experiments = []
    
    data_config = (
        DataPresets.blocks_world_standard() if domain == "blocks_world"
        else DataPresets.eight_puzzle_standard()
    )
    
    for size_name, size_fn in [
        ("tiny", ModelPresets.tiny),
        ("small", ModelPresets.small),
        ("medium", ModelPresets.medium),
        ("large", ModelPresets.large),
    ]:
        for use_wm in [False, True]:
            for weight_sharing in [False, True]:
                model_config = size_fn(
                    use_world_model=use_wm,
                    weight_sharing=weight_sharing
                )
                
                experiment = ExperimentConfig(
                    model=model_config,
                    data=data_config,
                    experiment_name=f"{domain}_{size_name}_{'wm' if use_wm else 'base'}_{'shared' if weight_sharing else 'std'}"
                )
                experiments.append(experiment)
    
    return experiments


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("EXAMPLE CONFIGURATIONS")
    print("=" * 70)
    
    print("\n1. Standard Blocks World experiment:")
    config = ExperimentConfig(
        model=ModelPresets.small(use_world_model=True),
        data=DataPresets.blocks_world_standard(),
        experiment_name="blocks_world_baseline"
    )
    print(config)
    
    print("\n2. Model size sweep:")
    sweep = create_model_size_sweep("blocks_world", SplitType.PRODUCTIVITY)
    print(f"Created {len(sweep)} experiments:")
    for exp in sweep[:3]:  # Show first 3
        print(f"  - {exp.experiment_name}")
    print(f"  ... and {len(sweep) - 3} more")
    
    print("\n3. Weight sharing ablation:")
    ws_sweep = create_weight_sharing_sweep("8_puzzle")
    print(f"Created {len(ws_sweep)} experiments:")
    for exp in ws_sweep:
        print(f"  - {exp.experiment_name}")
