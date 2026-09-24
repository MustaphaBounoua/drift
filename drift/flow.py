"""Stage 2: a conditional flow that transports z_r from a control cell to the perturbed state.

    t ~ U(0,1),  z_t = (1-t) z0 + t z1,  loss = || v(z_t, t | z_nr, u) - (z1 - z0) ||^2

z0 is the z_r of a control cell paired with the perturbed cell z1 by entropic OT. z_nr is carried
across unchanged. The stage-1 VAE is frozen; the perturbation encoder keeps training.
"""
from __future__ import annotations

import copy

import lightning as L
import numpy as np
import ot as pot
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from .config import FlowConfig
from .nets import DiTVelocityField


class LatentCache:
    """Posterior mean and sd of both latent blocks for every training cell, encoded once."""

    def __init__(self, vae, data, batch: int = 4096):
        s = data.train
        lab = np.asarray([str(c) for c in s.condition])
        dev = s.counts.device
        mu_r, sd_r, mu_nr, sd_nr = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(s.counts), batch):
                sl = slice(i, min(i + batch, len(s.counts)))
                o = vae.encode(s.counts[sl].float(), s.g1[sl], s.g2[sl], c=s.covar[sl], sample=False)
                mu_r.append(o["mu_r"]); sd_r.append((0.5 * o["lv_r"]).exp())
                mu_nr.append(o["mu_nr"]); sd_nr.append((0.5 * o["lv_nr"]).exp())
        self.mu_r, self.sd_r = torch.cat(mu_r), torch.cat(sd_r)
        self.mu_nr, self.sd_nr = torch.cat(mu_nr), torch.cat(sd_nr)
        self.g1, self.g2 = s.g1, s.g2
        self.rows = {c: torch.as_tensor(np.flatnonzero(lab == c), device=dev)
                     for c in sorted(set(lab))}
        self.ctrl = self.rows["control"]
        self.conds = [c for c in self.rows if c != "control"]

    def draw(self, idx):
        """Posterior samples (z_r, z_nr) at the given rows."""
        return (self.mu_r[idx] + self.sd_r[idx] * torch.randn_like(self.sd_r[idx]),
                self.mu_nr[idx] + self.sd_nr[idx] * torch.randn_like(self.sd_nr[idx]))


def _norm(C: torch.Tensor) -> torch.Tensor:
    """Scale a cost matrix to mean 1."""
    m = C.mean()
    return C / m if m > 0 else C


@torch.no_grad()
def build_pairs(cache: LatentCache, cfg: FlowConfig):
    """One round: sample conditions, then pair perturbed and control cells by OT on the latent means.

    Returns (perturbed rows, matched control rows).
    """
    p = cfg.pairing
    pick = torch.randperm(len(cache.conds))[:p.n_cond]
    P, Ctr = [], []
    for ci in pick.tolist():
        rows = cache.rows[cache.conds[ci]]
        n = min(p.block, len(rows))
        pr = rows[torch.randint(len(rows), (n,), device=rows.device)]
        cr = cache.ctrl[torch.randint(len(cache.ctrl), (n,), device=rows.device)]
        C = (p.alpha * _norm(torch.cdist(cache.mu_nr[pr], cache.mu_nr[cr]))
             + p.lam * _norm(torch.cdist(cache.mu_r[pr], cache.mu_r[cr])))
        a = np.ones(n) / n
        G = np.asarray(pot.sinkhorn(a, a, C.double().cpu().numpy(), p.ot_reg))
        plan = G / G.sum(1, keepdims=True).clip(1e-30)
        j = (plan.cumsum(1) > np.random.rand(len(plan))[:, None]).argmax(1)   # a partner per row
        P.append(pr); Ctr.append(cr[torch.as_tensor(j, device=rows.device)])
    return torch.cat(P), torch.cat(Ctr)


class PairStream(torch.utils.data.IterableDataset):
    """Endless batches of matched pairs; the OT pairing is re-solved every round."""

    def __init__(self, flow: "DriftFlow"):
        super().__init__()
        self.flow = flow

    def __iter__(self):
        bs = self.flow.cfg.train.flow_batch
        while True:
            pert, ctrl = build_pairs(self.flow.cache, self.flow.cfg)
            perm = torch.randperm(len(pert), device=pert.device)
            p, c = pert[perm], ctrl[perm]
            for i in range(0, len(p) - bs + 1, bs):
                yield dict(pert=p[i:i + bs], ctrl=c[i:i + bs])


class DriftFlow(L.LightningModule):

    def __init__(self, data, vae, cfg: FlowConfig | dict | None = None):
        super().__init__()
        cfg = FlowConfig() if cfg is None else (
            FlowConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg)
        self.save_hyperparameters({"cfg": cfg.to_dict()}, ignore=["data", "vae"])
        self.cfg, self.data = cfg, data
        self.vae = vae.eval().requires_grad_(False)
        self.pert_enc = vae.pert_enc
        self.pert_enc.requires_grad_(True).train()

        a, va = cfg.arch, vae.cfg.arch
        self.field = DiTVelocityField(va.d_r, va.d_nr, self.pert_enc.d_pert, a.hidden_dropout,
                                      d=a.cond_dim, n_layer=a.dit_layers, n_tok=a.dit_tokens)
        self.ema = copy.deepcopy(self.field).requires_grad_(False)
        self.fm = ConditionalFlowMatcher(sigma=0.0)
        self.cache: LatentCache | None = None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint["state_dict"]
        for k in ("z_mu", "z_sd", "vae._key2cls"):             # unused buffers of older checkpoints
            sd.pop(k, None)

    def setup(self, stage=None):
        if self.cache is None:
            self.cache = LatentCache(self.vae, self.data)

    def training_step(self, batch, batch_idx):
        pr, cr = batch["pert"], batch["ctrl"]
        zr_p, _ = self.cache.draw(pr)
        zr_c, znr_c = self.cache.draw(cr)
        t, xt, ut = self.fm.sample_location_and_conditional_flow(zr_c, zr_p)
        gt = self._condition_tokens(self.cache.g1[pr], self.cache.g2[pr])
        loss = ((self.field(xt, t, znr_c, gt) - ut) ** 2).mean()
        self.log("flow_loss", loss)
        return loss

    def on_train_batch_end(self, *_):
        d = self.cfg.train.ema_decay
        with torch.no_grad():
            for e, f in zip(self.ema.parameters(), self.field.parameters()):
                e.mul_(d).add_(f.detach(), alpha=1 - d)
            for e, f in zip(self.ema.buffers(), self.field.buffers()):
                e.copy_(f)

    def configure_optimizers(self):
        """Adam over the velocity field and the perturbation encoder."""
        params = list(self.field.parameters()) + list(self.pert_enc.parameters())
        return torch.optim.Adam(params, lr=self.cfg.train.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(PairStream(self), batch_size=None)

    def _condition_tokens(self, g1, g2):
        """Condition tokens of the two perturbed genes, with noise and dropout during training."""
        x = self.pert_enc.gene_tokens(g1, g2)
        if self.ema.training:
            if self.cfg.arch.pert_noise > 0:
                rms = x.detach().pow(2).mean().sqrt().clamp_min(1e-6)
                x = x + self.cfg.arch.pert_noise * rms * torch.randn_like(x)
            if self.cfg.arch.cond_out_dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.cfg.arch.cond_out_dropout, training=True)
        return x

    @torch.no_grad()
    def generate(self, g1_id: int, g2_id: int, n: int | None = None):
        """Predict n cells for one condition: transport sampled control cells with Euler steps."""
        n = n or self.cfg.train.n_gen
        dev = self.cache.mu_r.device
        cr = self.cache.ctrl[torch.randint(len(self.cache.ctrl), (n,), device=dev)]
        z, znr_c = self.cache.draw(cr)
        g1 = torch.full((n,), g1_id, device=dev, dtype=torch.long)
        g2 = torch.full((n,), g2_id, device=dev, dtype=torch.long)
        gt = self._condition_tokens(g1, g2)
        steps = self.cfg.train.ode_steps
        for k in range(steps):
            t = torch.full((n,), float(k / steps), device=dev)
            z = z + self.ema(z, t, znr_c, gt) / steps
        return self.vae.dec.to_eval_space(self.vae.dec(znr_c, z)).cpu().numpy()
