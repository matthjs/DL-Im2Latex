from loguru import logger
from peft import LoraConfig, IA3Config, get_peft_model
import warnings
from im2latex.conf.config_classes import Config
from im2latex.trainers import VisionEncoderDecoderTrainer

warnings.filterwarnings("ignore")


class VisionEncoderDecoderFinetuner(VisionEncoderDecoderTrainer):
    """
    Decoder class for VisionEncoderDecoderTrainer. Change the behavior to perform
    finetuning instead of full all parameter training.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.lora_config = None
        self.ia3_config = None

    def setup_finetuning(self) -> None:
        """
        Sets up finetuning configurations for LoRA and IA³ adapters.
        """
        # setting up the adapter
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                # Encoder (Swin Transformer) modules
                "attn.qkv",
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2",
                # Decoder (GPT-2) modules
                "c_attn",
                "c_proj",
                "c_fc",
                "attn.c_proj",
            ],
            lora_dropout=0.1,
            bias="none"
            # task_type="VL"  # Vision-Language task
        )

        # setting up the IA³ adapter
        # I do not recall this being used anywhere but have included it for now.
        self.ia3_config = IA3Config(
            target_modules=[
                # Encoder (Swin Transformer) modules
                "attn.qkv",
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2",
                # Decoder (GPT-2) modules
                "c_attn",
                "c_proj",
                "c_fc",
                "attn.c_proj",
            ],
            feedforward_modules=[
                "mlp.fc1",
                "mlp.fc2",
                "c_fc",
            ],
            init_ia3_weights=True,
            # task_type="VL"  # Vision-Language task
        )

        self.model = get_peft_model(self.model, self.lora_config)

        if self.cfg.log_level >= 1:
            logger.debug("Finetuning")
            self.model.print_trainable_parameters()

