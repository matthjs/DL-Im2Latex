"""
This wrapper class is needed to make sure Gradcam works.
"""
from typing import List

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Bring the channels to the first dimension,
# like in CNNs
def swin_reshape_transform(tensor, height=7, width=7):  # tensor is a tuple idk why
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


class SwinEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, feature_extractor):
        super().__init__()
        self.encoder = encoder
        self.feature_extractor = feature_extractor

    def forward(self, x):
        outputs = self.encoder(x)
        return outputs.last_hidden_state  # Return actual feature tensor


class GradCamAdaptor(GradCAM):
    def forward(
            self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss = torch.mean(loss)
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
