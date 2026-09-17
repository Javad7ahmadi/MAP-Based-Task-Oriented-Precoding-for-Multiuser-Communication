import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()

        self.margin = margin

    def forward(self, features, labels):

        # Pairwise squared Euclidean distances
        diff = features.unsqueeze(1) - features.unsqueeze(0)

        distances_sq = torch.sum(
            torch.abs(diff) ** 2,
            dim=2
        )

        # Same-class pairs: y = 1
        # Different-class pairs: y = 0
        labels = labels.unsqueeze(0)
        labels_t = labels.transpose(0, 1)

        same_class = (labels == labels_t).float()
        different_class = 1.0 - same_class

        # Avoid self-pairs
        batch_size = features.size(0)
        mask = ~torch.eye(
            batch_size,
            dtype=torch.bool,
            device=features.device
        )

        same_loss = same_class * distances_sq
        different_loss = different_class * torch.clamp(
            self.margin - torch.sqrt(
                distances_sq + 1e-8
            ),
            min=0.0
        ) ** 2

        loss = (same_loss + different_loss)[mask].mean()

        return loss