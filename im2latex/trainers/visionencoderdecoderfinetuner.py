from peft import LoraConfig, get_peft_model
import warnings
from im2latex.trainers import VisionEncoderDecoderTrainer
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from im2latex.conf.config_classes import Config
from datasets import load_dataset
from im2latex.util.latexdataset import LatexDataset, DataCollator
from transformers import VisionEncoderDecoderModel

warnings.filterwarnings("ignore")


class VisionEncoderDecoderFinetuner(VisionEncoderDecoderTrainer):
    """
    Decorator class for VisionEncoderDecoderTrainer. Change the behavior to perform
    finetuning instead of full parameter training.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.lora_config = None
        self.ia3_config = None
        self._setup_finetuning()

    def _setup_finetuning(self) -> None:
        """
        Sets up finetuning configurations for LoRA adapters.
        """
        if self.cfg.finetuning is None:
            raise ValueError("Hardcoded settings no supported, include finetuning"
                             "configs in the config file")

        print(f"Adding trainable parameters to {self.cfg.finetuning.target_modules}")

        # setting up the adapter
        self.lora_config = LoraConfig(
            r=self.cfg.finetuning.lora_r,
            lora_alpha=self.cfg.finetuning.lora_alpha,
            target_modules=list(self.cfg.finetuning.target_modules),
            lora_dropout=self.cfg.finetuning.lora_dropout,
            bias=self.cfg.finetuning.bias  # ignore type error
            # task_type="VL"  # Vision-Language task
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.setup_optimizers()

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

    def set_model_configs(self) -> None:
        if self.cfg.log_level == 1:
            logger.info(f"Skipping setting model configs when finetuning")

    def setup_dataset(self) -> None:
        """
        Code duplication w.r.t. parent class but for now this has to be done
        NOTE: This function is dataset specific.
        """
        dataset = load_dataset(self.cfg.dataset.train_dataset_path, self.cfg.dataset.split_dataset_name)
        train_ds = dataset['train']
        val_ds = dataset['validation']
        test_ds = dataset['test']

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