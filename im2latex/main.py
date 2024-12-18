import hydra
from omegaconf import OmegaConf
from loguru import logger

from im2latex.parser.config_classes import Config
from im2latex.parser.parser import get_args


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """
    Main function, loads config file.
    """
    if cfg.log_level == 1:
        logger.info(OmegaConf.to_yaml(cfg))

    # TODO
    if cfg.mode == "train":
        pass
    elif cfg.mode == "fine_tune":
        pass
    elif cfg.mode == "inference":
        pass
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    main()
