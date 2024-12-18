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
            image_size,    # <---
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
        latex_sequence = item['latex_formula']
        image = item['image']

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

