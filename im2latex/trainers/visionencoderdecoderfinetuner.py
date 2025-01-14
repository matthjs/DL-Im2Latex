import os
from peft import LoraConfig, IA3Config, get_peft_model
import warnings
from im2latex.trainers import VisionEncoderDecoderTrainer
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from im2latex.conf.config_classes import Config
from datasets import load_dataset
from im2latex.util.latexdataset import LatexDataset, DataCollator

from transformers import logging, VisionEncoderDecoderModel

warnings.filterwarnings("ignore")


class VisionEncoderDecoderFinetuner(VisionEncoderDecoderTrainer):
    """
    Decorator class for VisionEncoderDecoderTrainer. Change the behavior to perform
    finetuning instead of full all parameter training.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.lora_config = None
        self.ia3_config = None
        self._setup_finetuning()

    def _setup_finetuning(self) -> None:
        """
        Sets up finetuning configurations for LoRA and IA³ adapters.
        """
        if self.cfg.finetuning is None:
            raise ValueError("Hardcoded settings no supported, include finetuning"
                             "configs in the config file")

        # setting up the adapter
        self.lora_config = LoraConfig(
            r=self.cfg.finetuning.lora_r,
            lora_alpha=self.cfg.finetuning.lora_alpha,
            target_modules=self.cfg.finetuning.target_modules,
            lora_dropout=self.cfg.finetuning.lora_dropout,
            bias=self.cfg.finetuning.bias  # ignore type error
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

    def construct_vision_model(self) -> None:
        """
        Instead of loading a (pretrained) encoder and decoder we want to load
        a full visionencoderdecoder model here.
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.cfg.model.vision_encoder_decoder_name  # Implicit assumption that this is set in the config
        )

        if self.cfg.log_level == 1:
            logger.info(f"Loaded VisionEncoderDecoderModel {self.cfg.model.vision_encoder_decoder_name}")

    def setup_dataset(self) -> None:
        """
        Code duplication w.r.t. parent class but for now this has to be done
        as this filter_dataset function needs to be used (?).
        """
        dataset = load_dataset(self.cfg.dataset.train_dataset_path, self.cfg.dataset.split_dataset_name)
        train_val_split = dataset["train"].train_test_split(test_size=self.cfg.dataset.val_test_size,
                                                            seed=self.cfg.torch.seed)
        train_ds = train_val_split["train"]
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=self.cfg.torch.seed)
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]

        # TODO: Check if this is necessary.
        def filter_dataset(dataset):
            def is_valid_sample(sample):
                try:
                    image = sample['image']
                    latex = sample['text']
                    return image is not None and latex is not None and len(latex) > 0
                except:
                    return False

            return dataset.filter(is_valid_sample)

        train_ds = filter_dataset(train_ds)
        val_ds = filter_dataset(val_ds)

        image_size = self.cfg.image.image_size

        self.train_dataset = LatexDataset(train_ds,
                                          self.tokenizer,
                                          self.feature_extractor,
                                          phase='train',
                                          image_size=image_size)

        self.val_dataset = LatexDataset(val_ds, self.tokenizer,
                                        self.feature_extractor,
                                        phase='val',
                                        image_size=image_size)

        self.test_dataset = LatexDataset(test_ds,
                                         self.tokenizer,
                                         self.feature_extractor,
                                         phase='test',
                                         image_size=image_size)

        train_sampler = RandomSampler(self.train_dataset)
        val_sampler = SequentialSampler(self.val_dataset)
        test_sampler = SequentialSampler(self.test_dataset)

        DataCollator.padding_value = self.tokenizer.pad_token_id

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.cfg.training.batch_size_train,
                                           sampler=train_sampler,
                                           collate_fn=DataCollator.data_collator)

        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.cfg.training.batch_size_val,
                                         sampler=val_sampler,
                                         collate_fn=DataCollator.data_collator)

        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.cfg.training.batch_size_val,
                                          sampler=test_sampler,
                                          collate_fn=DataCollator.data_collator)

    def save_model(self, checkpoint_path: str) -> 'AbstractTrainer':
        """
        Save the current model and tokenizer.

        :param checkpoint_path: Path to save the model checkpoint.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.base_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        return self
