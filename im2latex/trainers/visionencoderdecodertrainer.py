from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from im2latex.conf.config_classes import Config
from im2latex.trainers.abstracttrainer import AbstractTrainer
import torch
import torch.distributed as dist
import evaluate
from transformers import (
    VisionEncoderDecoderModel,
    SwinConfig,
    GPT2Config,
    AutoTokenizer,
    AutoFeatureExtractor
)
from transformers import (
    VisionEncoderDecoderModel,
    SwinConfig,
    GPT2Config,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup
)

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
import time
from torch.optim import AdamW
from im2latex.util.latexdataset import LatexDataset, DataCollator


class VisionEncoderDecoderTrainer(AbstractTrainer):
    """
    A trainer class for training and evaluating a VisionEncoderDecoderModel.
    """

    def __init__(
            self,
            cfg: Config
    ):
        super().__init__()
        # TODO: This constructor is way too long and needs to be refactored
        self.cfg = cfg

        # Distributed setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
        dist.init_process_group(backend='nccl')
        self.ddp_rank = int(os.environ['RANK'])
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{self.ddp_local_rank}'
        torch.cuda.set_device(device)
        self.master_process = self.ddp_rank == 0

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.feature_extractor)

        # Where is this config used?
        self.encoder_config = SwinConfig.from_pretrained(cfg.model.encoder_name)
        self.decoder_config = GPT2Config.from_pretrained(cfg.model.decoder_name)

        # Set up te vision encoder decoder model.
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            cfg.model.encoder_name,
            cfg.model.decoder_name
        )

        # Setting up model configs including decoding parameters
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.model.config.max_length = 256
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        # TODO: Make configurable or/and probably want to change this.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        torch.compile(self.model)

        # Dataset
        dataset = load_dataset(cfg.dataset.train_dataset_path, cfg.dataset.split_dataset_name)
        train_val_split = dataset["train"].train_test_split(test_size=cfg.dataset.val_test_size, seed=cfg.torch.seed)
        train_ds = train_val_split["train"]
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=cfg.torch.seed)
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]

        # creating datasets and dataloader
        # I would like to move this outside this class to be make it less
        # specific to our dataset but for now this will do.
        self.train_dataset = LatexDataset(train_ds, self.tokenizer, self.feature_extractor, phase='train')
        self.val_dataset = LatexDataset(val_ds, self.tokenizer, self.feature_extractor, phase='val')
        self.test_dataset = LatexDataset(test_ds, self.tokenizer, self.feature_extractor, phase='test')

        # We want to support distributed or non-distributed case.
        if cfg.training.distributed:
            train_sampler = DistributedSampler(self.train_dataset)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset)  # or SequentialSampler
            val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
            test_sampler = torch.utils.data.SequentialSampler(self.test_dataset)

        DataCollator.padding_value = self.tokenizer.pad_token_id
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.training.batch_size_train,
                                           sampler=train_sampler,
                                           collate_fn=DataCollator.data_collator)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=cfg.training.batch_size_val,
                                         sampler=val_sampler,
                                         collate_fn=DataCollator.data_collator)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=cfg.training.batch_size_val,
                                          sampler=test_sampler,
                                          collate_fn=DataCollator.data_collator)

        # Hyperparameters
        self.learning_rate = cfg.training.learning_rate
        self.warmup_steps = cfg.training.warmup_steps
        self.num_epochs = cfg.training.num_epochs

        # Checkpointing
        self.eval_steps = cfg.checkpoint.eval_steps
        self.best_checkpoint_step = None
        self.checkpoint_dir = cfg.checkpoint.checkpoint_dir

        # Optimization
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            betas=cfg.training.betas,
            eps=cfg.training.eps)
        effective_batch_size = cfg.training.batch_size_train  # * ddp_world_size TODO <---
        max_steps = (len(self.train_dataset) // effective_batch_size) * cfg.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=max_steps
        )

        self.bleu_metric = evaluate.load(cfg.metric.bleu)

        if cfg.training.distributed:
            # setup of Distributed Data Parallel (DDP)
            model = DDP(self.model,
                        device_ids=[self.ddp_local_rank],
                        output_device=self.ddp_local_rank,
                        find_unused_parameters=False)

        # TODO: Make this configurable
        self.eval_max_batches = None

        # TODO: This is a bit akward
        self.avg_val_loss = None
        self.avg_bleu = None

    def train(self) -> 'AbstractTrainer':
        """
        Train the VisionEncoderDecoderModel.
        TODO: This method is too long
        """
        self.model.train()

        # TODO: Might want to have this as class methods.
        train_losses = []
        interval_losses = []
        best_val_loss = float('inf')
        global_step = 0  # TODO: ???

        total_steps_per_epoch = len(self.train_dataloader)

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            for step, batch in enumerate(
                    tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                         disable=self.ddp_local_rank != 0)):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                # averaging losses over gpus
                if self.cfg.training.distributed:
                    interval_loss_tensor = loss.clone().detach().to(self.device)
                    torch.distributed.all_reduce(interval_loss_tensor, op=torch.distributed.ReduceOp.AVG)
                    interval_losses.append(interval_loss_tensor.item())
                else:
                    interval_losses.append(loss.item())

                # Increment global_step
                global_step = epoch * total_steps_per_epoch + step + 1

                if global_step % self.eval_steps == 0 or (
                        epoch == self.num_epochs - 1 and step == total_steps_per_epoch - 1):
                    # computing the average loss for the last eval_steps
                    average_loss = np.mean(interval_losses)
                    train_losses.append(average_loss)

                    self.eval_max_batches = 20
                    self.evaluate()
                    val_loss, bleu_score = self.avg_val_loss, self.avg_bleu
                    self.eval_max_batches = None  # TODO: This could be better.

                    if self.master_process and self.cfg.log_level >= 1:
                        logger.info(
                            f"Step {global_step} - Train Loss: {average_loss}, Val Loss: {val_loss}, BLEU: {bleu_score}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # TODO: This does not have the same behavior as original code
                        # TODO: Fix/implement this.
                        self.save_model(os.path.join(self.checkpoint_dir, f"best_checkpoint_step_{global_step}"))

            torch.cuda.synchronize()  # TODO: Is this needed in distributed setting.
            epoch_duration = time.time() - epoch_start_time
            if self.master_process:
                if self.cfg.log_level >= 1:
                    logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.")

        return self

    def evaluate(self) -> 'AbstractTrainer':
        """
        Evaluate the VisionEncoderDecoderModel.
        TODO: REFACTOR THIS METHOD

        :return: A tuple containing validation loss and BLEU score.
        """
        self.model.eval()
        val_losses = []
        bleu_scores = []
        num_evaluated_batches = 0

        with (torch.no_grad()):
            effective_batch_size = self.cfg.training.batch_size_val * self.ddp_world_size
            max_steps = len(
                self.val_dataset) // effective_batch_size  # max over the whole eval, but we use only 20 batches (steps)

            if self.eval_max_batches is None:
                eval_iterator = tqdm(self.val_dataloader, desc=f"Evaluation", disable=self.ddp_local_rank != 0)
            else:
                eval_iterator = tqdm(self.val_dataloader, desc=f"Evaluation", total=self.eval_max_batches, disable=self.ddp_local_rank != 0)

            for batch_idx, batch in enumerate(eval_iterator):
                if self.eval_max_batches is not None and num_evaluated_batches >= self.eval_max_batches:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                if not self.cfg.training.distributed:
                    val_losses.append(loss.item())
                else:
                    step_loss_tensor = loss.clone().detach().to(self.device)
                    dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.AVG)
                    val_losses.append(step_loss_tensor.item())

                # Generate predictions
                # TODO: Make this not hardcoded?
                generated_ids = self.model.module.generate(pixel_values, num_beams=4, max_length=256, early_stopping=True)
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                label_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                bleu = self.bleu_metric.compute(predictions=generated_texts, references=label_texts)
                bleu_scores.append(bleu["google_bleu"])

                if self.cfg.training.distributed:
                    bleu_tensor = torch.tensor(bleu, device=self.device)
                    dist.all_reduce(bleu_tensor, op=dist.ReduceOp.AVG)
                    bleu_scores.append(bleu_tensor.item())
                else:
                    bleu_scores.append(bleu_tensor.item())

                num_evaluated_batches += 1

        self.avg_val_loss = np.mean(val_losses)
        self.avg_bleu = np.mean(bleu_scores)

        # Do something with these losses and scores
        return self

    def save_model(self, checkpoint_path: str) -> 'AbstractTrainer':
        """
        Save the current model and tokenizer.

        :param checkpoint_path: Path to save the model checkpoint.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        return self

    def load_model(self, checkpoint_path: str) -> 'AbstractTrainer':
        """
        Load a model and tokenizer from the checkpoint path.

        :param checkpoint_path: Path to the model checkpoint.
        """
        self.model = self.model.from_pretrained(checkpoint_path).to(self.device)
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")

        return self
