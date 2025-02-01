from abc import ABC, abstractmethod
import torch


def set_seed(seed: int) -> None:
    """
    Set seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self) -> 'AbstractTrainer':
        """
        Train a model.
        """
        pass

    @abstractmethod
    def evaluate(self) -> tuple:
        """
        Evaluate a model.
        """
        pass

    @abstractmethod
    def save_model(self, checkpoint_path: str) -> 'AbstractTrainer':
        pass

    @abstractmethod
    def load_model(self, checkpoint_path) -> 'AbstractTrainer':
        pass
