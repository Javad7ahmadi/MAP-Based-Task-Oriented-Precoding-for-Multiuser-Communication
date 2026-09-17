import torch


def Q(x):
    """
    Gaussian Q-function.
    """
    return 0.5 * torch.erfc(
        x / torch.sqrt(
            torch.tensor(2.0, device=x.device)
        )
    )


def loss(features, labels, priors,
         beta_min,
         beta_max):

    """
    MAP error surrogate under the equal-class-covariance assumption.

    Assumption:
        lambda_j = lambda_k = lambda

    Define:
        beta = lambda / R

    Therefore, beta is sampled from:
        [beta_min, beta_max]

    features:
        Complex features, shape [B, D]

    labels:
        Class labels, shape [B]

    priors:
        Class prior probabilities, shape [C]

    R:
        Received-signal dimension.
    """

    device = features.device

    # ---------------------------------------------------------
    # Randomly select beta for this mini-batch
    # ---------------------------------------------------------

    beta = (
        beta_min
        + (beta_max - beta_min)
        * torch.rand(1, device=device)
    )

    # ---------------------------------------------------------
    # Classes
    # ---------------------------------------------------------

    classes = torch.unique(labels)

    # ---------------------------------------------------------
    # Calculate class means
    # ---------------------------------------------------------

    means = {}

    for c in classes:

        c = c.item()

        mask = labels == c
        x = features[mask]

        mu = torch.mean(x, dim=0)

        means[c] = mu

    # ---------------------------------------------------------
    # Pairwise MAP error
    # ---------------------------------------------------------

    total_loss = torch.tensor(
        0.0,
        device=device
    )

    num_pairs = 0

    eps = 1e-8

    for j in classes:

        j = j.item()

        for k in classes:

            k = k.item()

            if j == k:
                continue

            mu_j = means[j]
            mu_k = means[k]

            p_j = priors[j]
            p_k = priors[k]

            # -------------------------------------------------
            # Mean difference
            # -------------------------------------------------

            mu_jk = mu_j - mu_k

            mu_distance = torch.sum(
                torch.abs(mu_jk) ** 2
            )

            # -------------------------------------------------
            # Q-function numerator
            #
            # ||mu_jk||^2 / beta
            # + log(p_j / p_k)
            # -------------------------------------------------

            numerator = (
                mu_distance / (beta + eps)
                +
                torch.log(
                    (p_j + eps) /
                    (p_k + eps)
                )
            )

            # -------------------------------------------------
            # Q-function denominator
            #
            # sqrt(
            #     2 ||mu_jk||^2 / beta
            # )
            # -------------------------------------------------

            denominator = torch.sqrt(
                2.0
                * mu_distance
                / (beta + eps)
                + eps
            )

            # -------------------------------------------------
            # Q-function argument
            # -------------------------------------------------

            argument = numerator / denominator

            pairwise_error = Q(argument)

            # -------------------------------------------------
            # Weighted pairwise error
            # -------------------------------------------------

            total_loss = (
                total_loss
                + p_j * pairwise_error
            )

            num_pairs += 1

    # ---------------------------------------------------------
    # Return loss
    # ---------------------------------------------------------

    if num_pairs == 0:

        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True
        )

    return total_loss