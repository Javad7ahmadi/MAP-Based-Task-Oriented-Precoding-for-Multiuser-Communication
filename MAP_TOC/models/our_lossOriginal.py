import torch
import torch.nn as nn

class OurLoss(nn.Module):

    def __init__(self, center_weight=0.0):
        super().__init__()
        self.center_weight = center_weight

    def forward(self, Z, y):

        classes = torch.unique(y)
        C = len(classes)
        D = Z.shape[1]

        mus = []
        deltas = []

        # =========================
        # compute class statistics
        # =========================
        for c in classes:

            Zc = Z[y == c]

            mu = Zc.mean(dim=0)
            mus.append(mu)

            delta = torch.sum(torch.abs(Zc - mu) ** 2) / (D * Zc.shape[0] + 1e-8)
            deltas.append(delta)

        mus = torch.stack(mus)
        deltas = torch.stack(deltas)

        # =========================
        # global variance term
        # =========================
        delta_max = torch.max(deltas)

        loss = 0.0

        # =========================
        # inter-class separation term
        # =========================
        for j in range(C):
            for k in range(j):
                if j == k:
                    continue

                dist2 = torch.sum(torch.abs(mus[j] - mus[k]) ** 2)

                denom = torch.sqrt(delta_max + .5)

                argument = torch.sqrt(dist2) / denom

                Q = 0.5 * torch.erfc(
                    argument / torch.sqrt(torch.tensor(2.0, device=Z.device))
                )

                loss = loss + Q

        return loss, {
            "delta_max": delta_max.item(),
            "mean_delta": deltas.mean().item()
        }