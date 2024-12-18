from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
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

from transformers import logging

# Stop intialization of weights warning
logging.set_verbosity_error()


class VisionEncoderDecoderTrainer(AbstractTrainer):
    """
    A trainer class for training and evaluating a VisionEncoderDecoderModel.
    """
    
    def setup_model(self):
        """
        Set up the VisionEncoderDecoderModel and tokenizer
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.cfg.model.feature_extractor)
        
        # Set up te vision encoder decoder model.
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            self.cfg.model.encoder_name,
            self.cfg.model.decoder_name
        )
        
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.model.config.no_repeat_ngram_size = self.cfg.decoding.no_repeat_ngram_size
        self.model.config.length_penalty = self.cfg.decoding.length_penalty
        
        self.max_length = self.cfg.decoding.max_length
        self.early_stopping = self.cfg.decoding.early_stopping
        self.num_beams = self.cfg.decoding.num_beams
        
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        torch.compile(self.model)
        
    def setup_dataset(self):
        
        dataset = load_dataset(self.cfg.dataset.train_dataset_path, self.cfg.dataset.split_dataset_name)
        train_val_split = dataset["train"].train_test_split(test_size=self.cfg.dataset.val_test_size, seed=self.cfg.torch.seed)
        train_ds = train_val_split["train"]
        val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=self.cfg.torch.seed)
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]
        
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
        
    def setup_optimizers(self):
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            betas=self.cfg.training.betas,
            eps=self.cfg.training.eps
        )

        effective_batch_size = self.cfg.training.batch_size_train
        max_steps = (len(self.train_dataset) // effective_batch_size) * self.cfg.training.num_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.cfg.training.warmup_steps,
            num_training_steps=max_steps
        )

    def __init__(
            self,
            cfg: Config
    ):
        super().__init__()
        #! Current setup assumes a very specific order of initialization.
        self.cfg = cfg

        # Assigns variables for model, tokenizer, feature_extractor
        self.setup_model()

        # Assigns variables for train, val, test datasets and dataloaders
        self.setup_dataset()

        # Hyperparameters
        self.learning_rate = cfg.training.learning_rate
        self.warmup_steps = cfg.training.warmup_steps
        self.num_epochs = cfg.training.num_epochs

        # Checkpointing
        self.eval_steps = cfg.checkpoint.eval_steps
        self.best_checkpoint_step = None
        self.checkpoint_dir = cfg.checkpoint.checkpoint_dir

        # Assigns variables for optimizer and scheduler
        self.setup_optimizers()

        self.bleu_metric = evaluate.load(cfg.metric.bleu)

        self.eval_max_batches = cfg.training.eval_max_batches

        # Store average validation loss and BLEU score for potential future reference
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
        total_steps = 0

        total_steps_per_epoch = len(self.train_dataloader)

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            for step, batch in enumerate(
                    tqdm(self.train_dataloader, 
                         desc=f"Epoch {epoch + 1}/{self.num_epochs}"
                         )
                    ):
                
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

                interval_losses.append(loss.item())

                # Increment global_step
                total_steps = epoch * total_steps_per_epoch + step + 1

                if total_steps % self.eval_steps == 0 or (
                    epoch == self.num_epochs - 1 and step == total_steps_per_epoch - 1):
                    
                    # computing the average loss for the last eval_steps
                    average_loss = np.mean(interval_losses)
                    train_losses.append(average_loss)

                    val_loss, bleu_score = self.evaluate()

                    if self.cfg.log_level >= 1:
                        logger.info(
                            f"Step {total_steps} - Train Loss: {average_loss}, Val Loss: {val_loss}, BLEU: {bleu_score}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(os.path.join(self.checkpoint_dir, f"best_checkpoint_step_{total_steps}"))

            epoch_duration = time.time() - epoch_start_time
            
            if self.cfg.log_level >= 1:
                logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.")

        return self

    def evaluate(self) -> 'AbstractTrainer':
        """
        Evaluate the VisionEncoderDecoderModel.

        :return: A tuple containing validation loss and BLEU score.
        """
        self.model.eval()
        val_losses = []
        bleu_scores = []
        num_evaluated_batches = 0

        with (torch.no_grad()):

            eval_iterator = tqdm(self.val_dataloader, desc=f"Evaluation", total=self.eval_max_batches)

            for batch_idx, batch in enumerate(eval_iterator):
                if num_evaluated_batches >= self.eval_max_batches:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(pixel_values=pixel_values, labels=labels)              
                val_losses.append(outputs.loss.item())

                # Generate predictions
                generated_ids = self.model.module.generate(pixel_values,
                                                           num_beams=self.num_beams,
                                                           max_length=self.max_length,
                                                           early_stopping=True)
                
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                label_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                bleu = self.bleu_metric.compute(predictions=generated_texts, references=label_texts)
                bleu_scores.append(bleu["google_bleu"])

                bleu_tensor = torch.tensor(bleu, device=self.device)
                bleu_scores.append(bleu_tensor.item())

                num_evaluated_batches += 1

        self.avg_val_loss = np.mean(val_losses)
        self.avg_bleu = np.mean(bleu_scores)

        # Return metrics
        return self.avg_val_loss, self.avg_bleu

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
