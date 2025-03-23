import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.decoder(x)
        return x

# Example usage
def get_model(load_weights=True, model_path="unet_trained.pth"):
    model = UNet()
    if load_weights and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model weights!")
    else:
        print("No trained model found, returning untrained model!")
    return model