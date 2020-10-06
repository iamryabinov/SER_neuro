import os
from torchsummary import summary

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url




class AlexNetMultiTask(nn.Module):

    def __init__(self, num_emotions=4, num_speakers=10, num_genders=2):
        super(AlexNetMultiTask, self).__init__()
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
        self.joint_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
        )
        self.classifier_emotion = nn.Sequential(
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_emotions),
        )
        self.classifier_speaker = nn.Sequential(
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_speakers),
        )
        self.classifier_gender = nn.Sequential(
            nn.Dropout(0.75),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_genders),
        )

    def forward(self, x):
        hidden = self.features(x)
        hidden = self.avgpool(hidden)
        hidden = hidden.flatten(1)
        hidden = self.joint_classifier(hidden)
        emotion_prediction = self.classifier_emotion(hidden)
        speaker_prediction = self.classifier_speaker(hidden)
        gender_prediction = self.classifier_gender(hidden)
        return emotion_prediction, speaker_prediction, gender_prediction


if __name__ == '__main__':
    model = AlexNetMultiTask()
    device = torch.device('cpu')
    model = model.to(device)
    x = torch.rand((1, 1, 224, 224))
    print(model(x))
    # summary(model, (1, 224, 224), device='cpu')