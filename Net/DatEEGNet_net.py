import torch
import torch.nn as nn
from layers import LinearWithConstraint, Conv2dWithConstraint


# Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class EEGNetWithDAT(nn.Module):
    def __init__(self, args, shape, num_domains=2):
        super(EEGNetWithDAT, self).__init__()
        self.num_ch = shape[2]
        self.F1 = 16
        self.F2 = 32
        self.D = 2
        self.sr = 250
        self.P1 = 4
        self.P2 = 8
        self.t1 = 16

        # Temporal Conv (EEGNet 1st block)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.F1, kernel_size=(1, self.sr // 2), bias=False, padding="same"), nn.BatchNorm2d(self.F1)
        )
        # Spatial Conv (EEGNet 2nd block)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.F1, self.F1 * self.D, kernel_size=(self.num_ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P1)),
        )
        # Separable Conv (EEGNet 3rd block)
        self.separable_conv = nn.Sequential(
            # Depth-wise
            nn.Conv2d(self.F1 * self.D, self.F2, kernel_size=(1, self.t1), groups=self.F1 * self.D, bias=False),
            # Point-wise
            nn.Conv2d(self.F2, self.F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.P2)),
        )
        # Task Classifier
        self.task_classifier = nn.Sequential(
            nn.Flatten(), LinearWithConstraint(in_features=self.F2 * 33, out_features=4, max_norm=0.25)
        )
        # Domain Discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Flatten(), nn.Linear(self.F2 * 33, 128), nn.ReLU(), nn.Linear(128, num_domains), nn.Softmax(dim=1)
        )

    def forward(self, x, lambda_=0.1):
        # Shared Feature Extractor
        features = self.temporal_conv(x)
        features = self.spatial_conv(features)
        features = self.separable_conv(features)

        # Task Prediction
        task_output = self.task_classifier(features)

        # Domain Prediction
        reversed_features = GradientReversal.apply(features, lambda_)
        domain_output = self.domain_discriminator(reversed_features)

        return task_output, domain_output
