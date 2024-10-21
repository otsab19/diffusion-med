import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLossVGG16(nn.Module):
    def __init__(self, model_path=None, layers=None, use_l1=True):
        super(PerceptualLossVGG16, self).__init__()
        # Load the VGG model from a given path or use a pre-trained model
        if model_path is not None:
            vgg = models.vgg16()
            vgg.load_state_dict(torch.load(model_path))  # Load model weights from the provided path
            vgg = vgg.features  # We only need the feature layers
        else:
            vgg = models.vgg16(pretrained=True).features

        # Freeze VGG16 parameters
        for param in vgg.parameters():
            param.requires_grad = False

        # Specify the layers to use for perceptual loss
        if layers is None:
            self.layers = [4, 9, 16, 23]  # conv1_2, conv2_2, conv3_3, conv4_3
        else:
            self.layers = layers

        # Create a sequential model with selected layers
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(self.layers) + 1)])
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, x, target):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            target = layer(target)
            if i in self.layers:
                loss += self.criterion(x, target)
        return loss
