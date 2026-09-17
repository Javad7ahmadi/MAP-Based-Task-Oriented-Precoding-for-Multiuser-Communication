import torch
import torch.nn as nn


class MaximalCodingRateReduction(nn.Module):

    def __init__(
        self,
        gam1=1.0,
        gam2=1.0,
        eps=0.01
    ):

        super().__init__()

        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    # ------------------------------------------------------------
    # Log determinant
    # ------------------------------------------------------------

    def logdet(self, X):

        sign, logdet = torch.linalg.slogdet(X)

        return sign * logdet

    # ------------------------------------------------------------
    # Discriminative loss
    # ------------------------------------------------------------

    def compute_discriminative_loss(
        self,
        W
    ):

        p, m = W.shape

        I = torch.eye(
            p,
            dtype=W.dtype,
            device=W.device
        )

        scalar = (
            p /
            (m * self.eps)
        )

        X = (
            I +
            self.gam1 *
            scalar *
            W.matmul(W.conj().T)
        )

        logdet = self.logdet(X)

        return torch.real(logdet) / 2.0

    # ------------------------------------------------------------
    # Compressive loss
    # ------------------------------------------------------------

    def compute_compressive_loss(
        self,
        W,
        labels
    ):

        p, m = W.shape

        classes = torch.unique(labels)

        I = torch.eye(
            p,
            dtype=W.dtype,
            device=W.device
        )

        compress_loss = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=W.device
        )

        for c in classes:

            mask = (
                labels == c
            )

            W_c = W[:, mask]

            n_c = W_c.shape[1]

            if n_c == 0:
                continue

            scalar = (
                p /
                (n_c * self.eps)
            )

            X = (
                I +
                scalar *
                W_c.matmul(
                    W_c.conj().T
                )
            )

            logdet = self.logdet(X)

            compress_loss += (
                torch.real(logdet) *
                n_c /
                m
            )

        return compress_loss / 2.0

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(
        self,
        X,
        labels
    ):

        # X: [B, D]
        #
        # MCR2 uses W: [D, B]

        W = X.T

        discriminative_loss = (
            self.compute_discriminative_loss(
                W
            )
        )

        compressive_loss = (
            self.compute_compressive_loss(
                W,
                labels
            )
        )

        total_loss = (
            -self.gam2 *
            discriminative_loss
            +
            compressive_loss
        )

        return total_loss