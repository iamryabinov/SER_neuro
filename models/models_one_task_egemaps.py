import os
from torchsummary import summary

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class AlexNetEgemaps2048(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNetEgemaps2048, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(inplace=True),
        )
        self.ln1 = nn.LayerNorm(normalized_shape=2048 + 88)
        self.fc2 = nn.Sequential(
            nn.Linear(2048 + 88, 512),
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, spectrogram, egemaps):
        x1 = spectrogram
        x2 = egemaps
        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = self.fc1(x1)

        x = torch.cat((x1, x2), dim=1)
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x
