import torch
import torch.nn as nn
import torchvision


class MobileNetClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.lm_head = nn.Linear(1000, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        logits = self.model(x)
        logits = self.lm_head(logits)
        probs = self.activation(logits)
        return probs.squeeze()