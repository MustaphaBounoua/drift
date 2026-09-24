"""Networks.

    AdditivePertEncoder   perturbation -> e_u, and the prior p(z_r | u)
    Encoder               q(z_nr, z_r | x, u, c)
    Decoder               (z_nr, z_r) -> log-normalised expression
    CovariatePrior        p(z_nr | c)
    DiTVelocityField      stage-2 velocity field
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

TARGET_SUM_ENC = 1.0e4          # library size the encoder input is normalised to


def _mlp(*dims: int, out: int | None = None) -> nn.Sequential:
    """Linear-LayerNorm-GELU stack; `out` appends a final Linear."""
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    if out is not None:
        layers += [nn.Linear(dims[-1], out)]
    return nn.Sequential(*layers)


def _gauss(mu: torch.Tensor, lv: torch.Tensor, sample: bool) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu


class AdditivePertEncoder(nn.Module):
    """s0 = phi(g1) + phi(g2);  s = s0 + psi([s0, phi(g1)*phi(g2)]);  e_u = rho(s).

    psi is zero-initialised, so the composition starts additive. Gene id `n_genes` means no gene.
    """

    def __init__(self, W: torch.Tensor, d_pert: int, d_r: int, hidden: int = 256, noise: float = 0.0):
        super().__init__()
        self.register_buffer("W", W)                      # (n_genes, d_feat), fixed
        self.phi = _mlp(W.shape[1], hidden, out=d_pert)
        self.psi = _mlp(2 * d_pert, hidden, out=d_pert)
        nn.init.zeros_(self.psi[-1].weight); nn.init.zeros_(self.psi[-1].bias)
        self.rho = _mlp(d_pert, hidden, out=d_pert)
        # `null` codes the control; `unknown` replaces genes that have no features.
        self.null = nn.Parameter(torch.zeros(d_pert))
        self.unknown = nn.Parameter(torch.randn(d_pert) * 0.02)
        self.head = nn.Linear(d_pert, 2 * d_r)            # p(z_r|u)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.n_genes, self.d_pert, self.noise = W.shape[0], d_pert, float(noise)
        undoc = W.abs().sum(1) == 0
        if bool(undoc.any()):
            self.register_buffer("undoc", undoc)
        else:
            self.undoc = None

    def _phi(self, g):
        Wsafe = torch.cat([self.W, self.W.new_zeros(1, self.W.shape[1])], 0)
        e = self.phi(Wsafe[g])
        if self.undoc is not None:
            m = torch.cat([self.undoc, self.undoc.new_zeros(1)], 0)[g].unsqueeze(-1)
            e = torch.where(m, self.unknown.to(e.dtype).expand_as(e), e)
        if self.training and self.noise > 0:
            # scaled by the batch RMS of the embedding
            e = e + self.noise * e.detach().pow(2).mean().sqrt().clamp_min(1e-6) * torch.randn_like(e)
        return e

    def forward(self, g1, g2):
        m1, m2 = (g1 != self.n_genes), (g2 != self.n_genes)
        p1, p2 = self._phi(g1) * m1.unsqueeze(-1), self._phi(g2) * m2.unsqueeze(-1)
        s = p1 + p2
        s = s + (m1 & m2).unsqueeze(-1).float() * self.psi(torch.cat([s, p1 * p2], -1))
        e = self.rho(s)
        is_ctrl = (~m1 & ~m2).unsqueeze(-1).float()
        return is_ctrl * self.null + (1 - is_ctrl) * e

    def prior_params(self, g1, g2):
        """(mu, logvar) of p(z_r | u)."""
        mu, lv = self.head(self(g1, g2)).chunk(2, -1)
        return mu, lv

    def gene_tokens(self, g1, g2) -> torch.Tensor:
        """The two perturbed genes as separate tokens, (B, 2*d_pert); an absent gene gets `null`."""
        out = []
        for g in (g1, g2):
            m = (g != self.n_genes).unsqueeze(-1).float()
            out.append(m * self._phi(g) + (1 - m) * self.null)
        return torch.cat(out, -1)


class CovariatePrior(nn.Module):
    """p(z_nr | c) = N(mu(c), diag(exp(lv(c))))."""

    def __init__(self, d_covar: int, d_nr: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_covar, hidden), nn.GELU(), nn.Linear(hidden, 2 * d_nr))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        mu, lv = self.net(c).chunk(2, -1)
        return mu, lv.clamp(-4, 3)


class Encoder(nn.Module):
    """q(z_nr, z_r | x, u, c): a shared trunk on [x, u, c] and one linear head per block."""

    def __init__(self, n_genes: int, d_nr: int, d_r: int, d_feat: int, d_covar: int,
                 hidden: int = 1024):
        super().__init__()
        self.d_feat, self.d_covar = d_feat, d_covar
        self.trunk = _mlp(n_genes + d_feat + d_covar, hidden, hidden)
        self.head_nr = nn.Linear(hidden, 2 * d_nr)
        self.head_r = nn.Linear(hidden, 2 * d_r)

    def forward(self, x, u=None, c=None, sample: bool = True):
        xin = torch.log1p(x / x.sum(-1, keepdim=True).clamp_min(1.0) * TARGET_SUM_ENC)
        u = xin.new_zeros(xin.shape[0], self.d_feat) if u is None else u
        parts = [xin, u] + ([c] if self.d_covar > 0 else [])
        h = self.trunk(torch.cat(parts, -1))
        mu_nr, lv_nr = self.head_nr(h).chunk(2, -1)
        mu_r, lv_r = self.head_r(h).chunk(2, -1)
        lv_nr, lv_r = lv_nr.clamp(-8, 4), lv_r.clamp(-8, 4)
        return dict(z_nr=_gauss(mu_nr, lv_nr, sample), z_r=_gauss(mu_r, lv_r, sample),
                    mu_nr=mu_nr, lv_nr=lv_nr, mu_r=mu_r, lv_r=lv_r)


class Decoder(nn.Module):
    """(z_nr, z_r) -> log-normalised expression, initialised to the mean control profile."""

    def __init__(self, n_genes: int, d_nr: int, d_r: int, hidden: int, init_bias: torch.Tensor):
        super().__init__()
        self.net = _mlp(d_nr + d_r, hidden, hidden, out=n_genes)
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(init_bias)

    def forward(self, z_nr, z_r):
        return self.net(torch.cat([z_nr, z_r], -1))

    @staticmethod
    def to_eval_space(mu: torch.Tensor) -> torch.Tensor:
        """Clamp to the valid range of log-normalised expression."""
        return mu.clamp_min(0.0)


class DiTVelocityField(nn.Module):
    """A transformer over z_r tokens, conditioned in context on [t, z_nr, gene1, gene2] tokens."""

    def __init__(self, d_r: int, d_nr: int, d_pert: int, dropout: float,
                 d: int = 256, n_layer: int = 6, n_head: int = 8, n_tok: int = 16):
        super().__init__()
        self.d, self.n_tok = d, n_tok
        self.to_tok = nn.Linear(d_r, n_tok * d)
        self.tok_norm = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.znr_proj = nn.Linear(d_nr, d)
        self.gene_proj = nn.Linear(d_pert, d)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=dropout,
                                       batch_first=True, activation="gelu", norm_first=True), n_layer)
        self.out = nn.Linear(n_tok * d, d_r)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _t_embed(self, t):
        half = self.d // 2
        f = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = (t.reshape(-1, 1).float() * 10000.0) * f.reshape(1, -1)
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, z_t, t, z_nr, gene_tokens):
        B = z_t.shape[0]
        x = self.tok_norm(self.to_tok(z_t).view(B, self.n_tok, self.d)) + self.pos.unsqueeze(0)
        g = self.gene_proj(gene_tokens.view(B, 2, -1))
        cond = [self._t_embed(t).unsqueeze(1), self.znr_proj(z_nr).unsqueeze(1), g]
        h = self.blocks(torch.cat(cond + [x], 1))
        return self.out(h[:, 4:].reshape(B, -1))
