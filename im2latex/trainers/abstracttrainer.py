from abc import ABC, abstractmethod
import torch


# Putting this function here for now but this can be moved.
def set_seed(seed: int) -> None:
    """
    Set seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AbstractTrainer(ABC):
    """
    I do not think it is necessary to have a very deep class hierarchy so maybe something like
    VisionEncoderDecoderTrainer --> AbstractTrainer is enough.
    You can also refactor and apply command pattern later if the trainer class becomes too big.
    """

    def __init__(self):
        """
        TODO: Add suitable fields.
        """

    @abstractmethod
    def train(self) -> 'AbstractTrainer':
        """
        Train a model.
        Maybe change signature to return AbstractTrainer for method
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

    # TODO Add other methods if needed.
