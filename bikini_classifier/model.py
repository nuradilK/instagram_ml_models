import torch.nn as nn
import torchvision


class MobileNetClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.mobilenet_v3_large(pretrained=True)
        # self.base = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        logits = self.base(x)
        logits = self.fc(logits)
        probs = self.activation(logits)
        return probs.squeeze()