import torch
import torch.nn as nn
from guided_diffusion.vgg import get_vgg_model

class PerceptualLossVGG16(nn.Module):
    def __init__(self, model_path=None, use_l1=True):
        super(PerceptualLossVGG16, self).__init__()

        # Load VGG16 model with grayscale input handled in get_vgg_model
        vgg = get_vgg_model()

        # Load weights from the specified model path, if provided
        if model_path:
            vgg.load_state_dict(torch.load(model_path), strict=False)

        # Extract the specific layers for perceptual loss computation
        self.layer_ids = [4, 9, 16, 23]  # Corresponding to conv1_2, conv2_2, conv3_3, conv4_3 layers in VGG16
        self.vgg = nn.Sequential(*[vgg.features[i] for i in range(max(self.layer_ids) + 1)])

        # Set the criterion (L1 or MSE loss) based on the input flag
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

        # Freeze VGG model weights since we do not train the perceptual loss network
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        """
        Compute the perceptual loss between x and target images by comparing feature maps
        from the specified layers of VGG16.

        Args:
            x (Tensor): Super-resolved image tensor.
            target (Tensor): High-resolution target image tensor.

        Returns:
            loss (Tensor): The perceptual loss value.
        """
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            target = layer(target)

            # Add the perceptual loss from the specified layers
            if i in self.layer_ids:
                loss += self.criterion(x, target)

        return loss
