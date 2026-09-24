"""Loss terms of stage 1.

    recon_mse       squared error summed over genes
    kl_normal       KL between diagonal Gaussians
    CLUBCont        CLUB upper bound on I(z_nr; e_u), the invariance penalty
    isometry_loss   relative isometry between condition embeddings and responses
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn



def recon_mse(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Squared error summed over genes, averaged over cells."""
    return ((x - mu) ** 2).sum(-1).mean()


def kl_normal(mu, logvar, prior_mu=None, prior_logvar=None) -> torch.Tensor:
    """KL(q || p) summed over dimensions, averaged over cells; p defaults to N(0, I)."""
    if prior_mu is None:
        per_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)
    else:
        per_dim = (0.5 * (prior_logvar - logvar
                          + (logvar.exp() + (mu - prior_mu) ** 2) / prior_logvar.exp() - 1.0)).mean(0)
    return per_dim.sum()


class CLUBCont(nn.Module):
    """CLUB upper bound on I(z_nr; e_u) with a Gaussian critic q(e_u | z_nr).

    e_u is reduced by a fixed random projection to `d_u` dimensions.
    """

    def __init__(self, d_z: int, d_pert: int, d_u: int = 16, hidden: int = 256):
        super().__init__()
        self.register_buffer("proj", torch.randn(d_pert, d_u) / math.sqrt(d_pert))
        self.net = nn.Sequential(nn.LayerNorm(d_z), nn.Linear(d_z, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, d_u)
        self.lv = nn.Sequential(nn.Linear(hidden, d_u), nn.Tanh())

    def _q(self, z):
        h = self.net(z)
        return self.mu(h), self.lv(h)

    def loglikeli(self, z, e_u):
        """Negative log-likelihood of the critic; minimised on detached inputs to fit it."""
        mu, lv = self._q(z)
        return (((mu - e_u @ self.proj) ** 2 / lv.exp() + lv).sum(-1)).mean()

    def mi_est(self, z, e_u):
        """E_p(z,u)[log q(u|z)] - E_p(z)p(u)[log q(u|z)]."""
        e = e_u @ self.proj
        mu, lv = self._q(z)
        var2 = 2.0 * lv.exp()
        pos = -(mu - e) ** 2 / var2
        neg = -((e.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(1) / var2
        return (pos.sum(-1) - neg.sum(-1)).mean()

def isometry_loss(F: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """1 - corr(pairwise distances of E, pairwise distances of F)."""
    if E.shape[0] < 3:
        return E.new_zeros(())
    DF, DE = torch.cdist(F, F), torch.cdist(E, E)
    iu = torch.triu_indices(E.shape[0], E.shape[0], offset=1, device=E.device)
    dF, dE = DF[iu[0], iu[1]], DE[iu[0], iu[1]]
    if float(dE.detach().std()) < 1e-8 or float(dF.detach().std()) < 1e-8:
        return E.new_ones(())
    r = (((dF - dF.mean()) * (dE - dE.mean())).mean()) / (dF.std() * dE.std()).clamp_min(1e-12)
    return 1.0 - r
