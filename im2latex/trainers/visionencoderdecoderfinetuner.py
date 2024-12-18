from im2latex.trainers.abstracttrainer import AbstractTrainer


class VisionEncoderDecoderFinetuner(AbstractTrainer):
    # TODO
    def train(self) -> None:
        pass

    def evaluate(self) -> 'AbstractTrainer':
        pass

    def save_model(self, checkpoint_path: str) -> 'AbstractTrainer':
        pass

    def load_model(self, checkpoint_path) -> 'AbstractTrainer':
        pass

