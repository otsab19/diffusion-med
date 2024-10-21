import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLossVGG16(nn.Module):
    def __init__(self, model_path=None, use_l1=True):
        super(PerceptualLossVGG16, self).__init__()
        vgg = models.vgg16(pretrained=False)  # Do not load pre-trained weights here
        if model_path:
            vgg.load_state_dict(torch.load(model_path), strict=False)  # Load weights with strict=False to allow for changes

        # Modify the first convolutional layer to accept 1-channel grayscale input
        old_weights = vgg.features[0].weight.data
        new_weights = old_weights.mean(dim=1, keepdim=True)  # Average over RGB channels to get grayscale equivalent
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        vgg.features[0].weight.data = new_weights

        # Extract the required feature layers for perceptual loss
        self.layers = [4, 9, 16, 23]  # Corresponding to conv1_2, conv2_2, conv3_3, conv4_3
        self.vgg = nn.Sequential(*[vgg.features[i] for i in range(max(self.layers) + 1)])

        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, x, target):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            target = layer(target)
            if i in self.layers:
                loss += self.criterion(x, target)
        return loss
