import torch


def loss(features, labels):
    """
    Discriminative Gain loss.

    Maximizes variance-normalized separation
    between class means.

    features: [B, D] complex
    labels:   [B]
    """

    device = features.device
    classes = torch.unique(labels)

    means = {}
    variances = {}

    # ------------------------------------------------------------
    # Class statistics
    # ------------------------------------------------------------

    for c in classes:

        c = c.item()

        x = features[labels == c]

        mu = torch.mean(x, dim=0)

        variance = torch.mean(
            torch.abs(x - mu) ** 2,
            dim=0
        )

        means[c] = mu
        variances[c] = variance

    # ------------------------------------------------------------
    # Pairwise DG
    # ------------------------------------------------------------

    total_dg = torch.tensor(
        0.0,
        device=device
    )

    num_pairs = 0

    eps = 1e-8

    for j in classes:

        j = j.item()

        for k in classes:

            k = k.item()

            if j >= k:
                continue

            mu_jk = means[j] - means[k]

            # Pooled variance
            d_jk = (
                variances[j] +
                variances[k]
            ) / 2.0

            dg_jk = torch.sum(
                torch.abs(mu_jk) ** 2 /
                (d_jk + eps)
            )

            total_dg = total_dg + dg_jk

            num_pairs += 1

    if num_pairs == 0:
        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True
        )

    # We maximize DG, therefore minimize -DG
    return -total_dg / num_pairs