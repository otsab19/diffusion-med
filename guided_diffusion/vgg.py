import torch.nn as nn
import torchvision.models as models

# Modify the VGG16 model to adapt for single-channel (grayscale) MRI data
def get_vgg_model(num_classes=1):
    vgg = models.vgg16(weights=None)  # Initialize without pre-trained weights
    # Modify the first convolutional layer to accept 1-channel grayscale input
    old_weights = vgg.features[0].weight.data
    new_weights = old_weights.mean(dim=1, keepdim=True)  # Average over 3 channels to create a single-channel conv
    vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Adjust for grayscale input
    vgg.features[0].weight.data = new_weights

    # Modify the classifier to output an image (224x224 instead of a class score)
    vgg.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 224 * 224),  # Output an image flattened (224 * 224)
        nn.Unflatten(1, (1, 224, 224))  # Reshape to [1, 224, 224]
    )

    return vgg
