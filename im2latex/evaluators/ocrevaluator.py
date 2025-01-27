from typing import List, Dict
from loguru import logger
import evaluate
import torch
import wandb
from torch import nn
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, SequentialSampler
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from im2latex.conf.config_classes import Config
from datasets import load_dataset
import numpy as np
from im2latex.evaluators.swingradcam import SwinEncoderWrapper, GradCamAdaptor, swin_reshape_transform
from im2latex.util.latexdataset import LatexDataset, DataCollator


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device


class OCREvaluator:
    def __init__(self, models: Dict[str, str], raw_dataset, image_size=(224, 468), batch_size=12):
        """
        Initialize the evaluator with the model, evaluation data, and metrics.

        :param model: VisionEncoderDecoderModel for OCR.
        """
        self.models = {}
        for model_name, model_path in models.items():
            print(model_name)
            print(model_path)
            self.models[model_name] = (
                VisionEncoderDecoderModel.from_pretrained(model_path).to("cuda"),
                AutoTokenizer.from_pretrained(model_path),
                AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
                # Hardcoded for now
            )

        self.dataset = raw_dataset
        self.metrics = {
            "google_bleu": evaluate.load("google_bleu"),
            # "perplexity": evaluate.load("perplexity", module_type="metric")
        }

        self.image_size = image_size  # Better way to handle this?
        self.batch_size = batch_size

        self.device = get_device()

    def evaluate(self, use_grad_cam: bool = True, grad_cam_batches: int = 1) -> dict:
        logger.info("Evaluate start")
        wandb.init(project="im2latex", name="evaluation-run")

        all_models_results = {}

        for model_name, comp in self.models.items():
            logger.info(f"Evaluating {model_name} on evaluation dataset...")
            model, tokenizer, feature_extractor = comp
            model.to(self.device)
            model.eval()

            wrapped_encoder = SwinEncoderWrapper(model.encoder, feature_extractor)
            target_layers = [model.encoder.encoder.layers[-1].blocks[-1].layernorm_after]

            DataCollator.padding_value = tokenizer.pad_token_id
            dataset = LatexDataset(self.dataset,
                                   tokenizer,
                                   feature_extractor,
                                   phase='test',
                                   image_size=self.image_size)
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    sampler=SequentialSampler(dataset),
                                    collate_fn=DataCollator.data_collator)

            metric_results = {}
            gradcam_visualizations = []

            eval_iterator = tqdm(dataloader, desc=f"Evaluation {model_name}")
            grad_cam_counter = 0
            for batch_idx, batch in enumerate(eval_iterator):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                metric_results["test loss"] = outputs.loss.item()

                generated_ids = model.generate(pixel_values)
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for metric_name, metric in self.metrics.items():
                    metric.add_batch(predictions=generated_texts, references=label_texts)

                if use_grad_cam and grad_cam_counter < grad_cam_batches:
                    logger.info("Executing gradcam")
                    grad_cam_counter += 1

                    with GradCamAdaptor(model=wrapped_encoder, target_layers=target_layers,
                                        reshape_transform=swin_reshape_transform) as cam:
                        grayscale_cam_batch = cam(input_tensor=pixel_values,
                                                  aug_smooth=True, eigen_smooth=True)

                        idx = 0
                        for grayscale_cam in grayscale_cam_batch:
                            # Convert to numpy and normalize to [0, 1]
                            # Apparently gradcam really wants this bruh
                            rgb_image = pixel_values[idx].cpu().numpy()
                            rgb_image = np.transpose(rgb_image, (1, 2, 0))
                            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
                            rgb_image = rgb_image.astype(np.float32)  # Ensure type is float32

                            visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
                            gradcam_visualizations.append(visualization)
                            idx += 1
                break
            for metric_name, metric in self.metrics.items():
                metric_results[metric_name] = metric.compute()
                # metric.reset()

            # Log GradCam results
            if use_grad_cam and gradcam_visualizations:
                from PIL import Image
                output_path = f"gradcam_visualization_{idx + 1}.png"
                Image.fromarray(visualization).save(output_path)
                print(f"GradCAM visualization saved to {output_path}")
                wandb.log({f"{model_name}_gradcam_visualizations": [wandb.Image(img) for img in
                                                                    gradcam_visualizations]})

            wandb.log({f"{model_name}_metrics": metric_results})
            all_models_results[model_name] = metric_results

        logger.info("Evaluation complete.")
        wandb.finish()

        return all_models_results
