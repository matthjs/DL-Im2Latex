from im2latex.trainers.abstracttrainer import AbstractTrainer


class VisionEncoderDecoderTrainer(AbstractTrainer):
    # TODO
    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def save_model(self, checkpoint_path: str) -> None:
        pass

    def load_model(self, checkpoint_path) -> None:
        pass