import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


# TODO: I copied this from the original code base for now but we will need to modify it.
class LatexDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            feature_extractor,
            phase,
            image_size,
            max_length=512
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def __len__(self):
        return len(self.dataset)

    def get_train_transform(self):
        def train_transform(image):
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image

        return train_transform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'latex_formula' in item:
            latex_sequence = item['latex_formula']
        elif 'text' in item:
            latex_sequence = item['text']
        else:
            raise ValueError("No valid dataset type does not have 'latex' or 'text' key for items")

        image = item['image']

        # converting RGBA to RGB for the test set --> some images have alphas
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # image processing
        if self.phase == 'train':
            img = self.train_transform(image)
            img = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img = image.resize(self.image_size)

        try:
            pixel_values = self.feature_extractor(
                images=img,
                return_tensors="pt",
            ).pixel_values.squeeze()
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            # providing a default tensor in case of error
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        latex_tokens = self.tokenizer(
            latex_sequence,
            padding=False,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'  # returning PyTorch tensors
        ).input_ids.squeeze()  # removing the batch dimension

        return {
            "pixel_values": pixel_values,
            "labels": latex_tokens
        }


class DataCollator:
    padding_value = -1

    @staticmethod
    # custom data collator
    def data_collator(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = [item['labels'] for item in batch]

        if DataCollator.padding_value < 0:
            raise ValueError("padding value not set.")

        # These first two conditions were only used in the finetuning in the original paper
        # but I do not see why they would case a problem with training on the standard dataset.
        if len(labels) == 0:
            # if all labels are empty, return a dummy tensor
            labels = torch.zeros((len(batch), 1), dtype=torch.long)
        elif len(labels) == 1:
            # if there's only one sample, add a dimension to make it a batch
            labels = labels[0].unsqueeze(0)
        else:
            # padding the labels
            labels = pad_sequence(labels, batch_first=True, padding_value=DataCollator.padding_value)

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }