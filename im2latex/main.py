import hydra
from omegaconf import OmegaConf
from loguru import logger
from im2latex.conf.config_classes import Config
from im2latex.trainers import VisionEncoderDecoderTrainer, VisionEncoderDecoderFinetuner
from dotenv import load_dotenv


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """
    Main function, loads config file.
    training --> python main.py --run --config-path conf --config-name config
    finetuning --> python main.py --run --config-path conf --config-name finetuning_config
    """
    if cfg.log_level == 1:
        logger.info(OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        trainer = VisionEncoderDecoderTrainer(cfg)
        trainer.train()
    elif cfg.mode == "finetuning":
        finetuner = VisionEncoderDecoderFinetuner(cfg)
        finetuner.train()
    elif cfg.mode == "inference":
        pass
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    load_dotenv()
    main()
