import torch
import torch.nn as nn


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feature_dim, device):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Complex class centers
        self.centers = nn.Parameter(
            torch.zeros(
                num_classes,
                feature_dim,
                dtype=torch.complex64,
                device=device
            )
        )

    def forward(self, features, labels):

        centers_batch = self.centers[labels]

        distances = torch.sum(
            torch.abs(
                features - centers_batch
            ) ** 2,
            dim=1
        )

        loss = torch.mean(distances)

        return loss