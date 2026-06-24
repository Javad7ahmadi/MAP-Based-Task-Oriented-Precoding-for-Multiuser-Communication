import torch
import torch.nn as nn
import torchvision.models as models


class MultiViewEncoder(nn.Module):
    def __init__(self, K=2, D_k=24, weights="IMAGENET1K_V1"):
        super().__init__()

        self.K = K
        self.D_k = D_k

        # separate backbone for each view
        self.backbones = nn.ModuleList([
            nn.Sequential(
                *list(models.resnet18(weights="IMAGENET1K_V1").children())[:-1]
            )
            for _ in range(K)
        ])

        # K separate projection heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, D_k)
            )
            )
            for _ in range(K)
        ])

    def forward(self, x_list):
        """
        x_list: list of K tensors
                each: [B, 3, H, W]
        """

        z_list = []

        for k in range(self.K):
            x = x_list[k]

            g = self.backbones[k](x)
            g = g.view(g.size(0), -1)  # [B, 512]

            z = self.heads[k](g)       # [B, D_k]

            z_list.append(z)

        # concatenate all device outputs
        Z = torch.cat(z_list, dim=1)  # [B, K*D_k]

        return Z, z_list