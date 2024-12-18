import argparse
from dataclasses import dataclass, field
from typing import Tuple
import hydra
from omegaconf import OmegaConf


# Default settings are from paper.
@dataclass
class TorchConfig:
    float32_matmul_precision: str = "high"
    seed: int = 42


@dataclass
class DatasetConfig:
    train_dataset_path: str = "OleehyO/latex-formulas"
    split_dataset_name: str = "cleaned_formulas"
    val_test_size: float = 0.2


@dataclass
class ModelConfig:
    tokenizer_name: str = "gpt2"
    feature_extractor: str = "microsoft/swin-base-patch4-window7-224-in22k"
    encoder_name: str = "microsoft/swin-base-patch4-window7-224-in22k"
    decoder_name: str = "gpt2"
    
@dataclass
class DecoderConfig:
    max_length: int = 256
    num_beams: int = 4
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    length_penalty: float = 2.0


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size_train: int = 8
    batch_size_val: int = 8
    eval_max_batches: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 400
    max_grad_norm: float = 1.0
    betas: Tuple[float, float] = (0.95, 0.98)
    eps: float = 1e-08


@dataclass
class ImageConfig:
    image_size: Tuple[int, int] = (224, 468)


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    eval_steps: int = 200


@dataclass
class MetricConfig:
    bleu: str = "google_bleu"


@dataclass
class Config:
    log_level: int
    torch: TorchConfig = field(default_factory=TorchConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
