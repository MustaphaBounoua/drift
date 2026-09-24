"""Stage 1: a VAE whose latent splits into an invariant block and a responsive one.

    z_nr ~ q(z_nr | x, u, c)    invariant, with prior p(z_nr | c)
    z_r  ~ q(z_r  | x, u, c)    responsive, with prior p(z_r | u)

    loss = [recon + beta_nr * KL_nr + beta_r * KL_r + lambda_club_u * CLUB(z_nr; e_u)] / n_genes
           + lambda_aux * aux + lambda_riso * riso
"""
from __future__ import annotations

from functools import partial

import lightning as L
import numpy as np
import torch

from .config import VAEConfig
from .data import _parse_genes
from .losses import CLUBCont, isometry_loss, kl_normal, recon_mse
from .nets import AdditivePertEncoder, CovariatePrior, Decoder, Encoder
from .ppi import cached_pathway, cached_ppi


class DriftVAE(L.LightningModule):

    def __init__(self, data, gene_features: torch.Tensor, cfg: VAEConfig | dict | None = None,
                 prior_table: torch.Tensor | None = None):
        super().__init__()
        cfg = VAEConfig() if cfg is None else (
            VAEConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg)
        self.save_hyperparameters({"cfg": cfg.to_dict()},
                                  ignore=["data", "gene_features", "prior_table"])
        self.cfg, self.data = cfg, data
        a, o, d = cfg.arch, cfg.objective, cfg.data
        self.steps = cfg.resolve(len(data.train.counts))
        G, C = data.train.expr.shape[1], data.n_covar

        # The encoder reads u from `gene_features`; p(z_r|u) reads the prior table, which appends
        # the Reactome and STRING blocks. `prior_table` is the complete table from a checkpoint.
        pf = gene_features
        if prior_table is not None:
            pf = prior_table
        else:
            if d.pathway_feat:
                pf = self._prior_block(data, pf, cached_pathway, d.pathway_feat)
            if d.ppi_feat:
                pf = self._prior_block(data, pf, partial(cached_ppi, min_score=d.ppi_min_score),
                                       d.ppi_feat)
        self.pert_enc = AdditivePertEncoder(pf, a.d_pert, a.d_r, hidden=a.pert_hidden,
                                            noise=a.pert_noise)
        self.register_buffer("enc_W", gene_features, persistent=False)

        self.enc = Encoder(G, a.d_nr, a.d_r, gene_features.shape[1], C if a.enc_covar else 0,
                           hidden=a.hidden)
        ctrl = data.train.expr[data.train.is_control].float().mean(0)
        self.dec = Decoder(G, a.d_nr, a.d_r, hidden=a.hidden, init_bias=ctrl)
        self.prior_nr = CovariatePrior(C, a.d_nr) if (C > 0 and a.prior_covar) else None

        self.club_u = CLUBCont(a.d_nr, a.d_pert, d_u=o.club_u_dim)
        self.club_opt = torch.optim.Adam(self.club_u.parameters(), lr=o.club_lr)
        if o.lambda_aux > 0:
            self.aux_head = torch.nn.Linear(a.d_pert, G)

    @staticmethod
    def _pert_gene_names(data) -> list[str]:
        """Every gene named by a train or test condition, sorted."""
        conds = {str(c) for c in data.train.condition} | {str(c) for c in data.test.condition}
        return sorted({g for c in conds if c != "control" for g in _parse_genes(c)})

    def _prior_block(self, data, pf, builder, width):
        """Append one gene-keyed feature block to the prior table."""
        names = self._pert_gene_names(data)
        Z = builder(names, width, device=pf.device)
        idx = {g: i for i, g in enumerate(names)}
        block = torch.zeros(pf.shape[0], Z.shape[1], device=pf.device)
        for gene, row in data.pert_gene_vocab.items():
            if gene in idx and 0 <= row < block.shape[0]:
                block[row] = Z[idx[gene]]
        return torch.cat([pf, block.to(pf.dtype)], 1)

    @classmethod
    def from_checkpoint(cls, path, data, map_location="cuda") -> "DriftVAE":
        """Load a trained VAE with the feature table stored in its checkpoint (`pert_enc.W`).

        ESM2, STRING and Reactome are not needed to load a trained model.
        """
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = VAEConfig.from_dict(ck["hyper_parameters"]["cfg"])
        W = ck["state_dict"]["pert_enc.W"].to(data.train.expr.device)
        return cls.load_from_checkpoint(path, data=data, gene_features=W[:, :cfg.data.d_feat].contiguous(),
                                        prior_table=W, map_location=map_location)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"].pop("_key2cls", None)       # unused buffer of older checkpoints

    # ---- encoding and priors --------------------------------------------------------------------
    def u_feature(self, g1, g2) -> torch.Tensor:
        """The encoder's view of u: feature rows summed over the perturbed genes."""
        W = self.enc_W
        Wsafe = torch.cat([W, W.new_zeros(1, W.shape[1])], 0)         # last row = no gene
        return Wsafe[g1] + Wsafe[g2]

    def encode(self, x, g1=None, g2=None, c=None, sample: bool = True) -> dict:
        """g1/g2 None means no perturbation."""
        u = self.u_feature(g1, g2) if g1 is not None else None
        return self.enc(x, u=u, c=c, sample=sample)

    @torch.no_grad()
    def encode_all(self, x, c=None, chunk: int = 2048) -> dict:
        """Sampled latents of unperturbed cells, in chunks."""
        outs = [self.encode(x[i:i + chunk], c=None if c is None else c[i:i + chunk])
                for i in range(0, len(x), chunk)]
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def prior_invariant(self, c):
        """p(z_nr|c); N(0,I) when there are no covariates."""
        if self.prior_nr is None:
            z = torch.zeros(c.shape[0], self.cfg.arch.d_nr, device=c.device)
            return z, torch.zeros_like(z)
        return self.prior_nr(c)

    # ---- training -------------------------------------------------------------------------------
    def _warmups(self):
        """Linear ramps for the KL terms and for the MI and conditioning terms."""
        def ramp(n):
            return min(1.0, (self.global_step + 1) / n) if n > 0 else 1.0
        return ramp(self.steps["kl_warmup_steps"]), ramp(self.steps["mi_warmup_steps"])

    def training_step(self, batch, batch_idx):
        o = self.cfg.objective
        x, c, g1, g2 = batch["counts"].float(), batch["covar"], batch["g1"], batch["g2"]
        s = self.encode(x, g1, g2, c=c, sample=True)

        recon = recon_mse(batch["expr"].float(), self.dec(s["z_nr"], s["z_r"]))
        mu_c, lv_c = self.prior_invariant(c)
        kl_nr = kl_normal(s["mu_nr"], s["lv_nr"], mu_c, lv_c)
        mu_p, lv_p = self.pert_enc.prior_params(g1, g2)
        kl_r = kl_normal(s["mu_r"], s["lv_r"], mu_p, lv_p)

        club = self._club_step(s["z_nr"], g1, g2)
        aux = self._aux_step(g1, g2)
        riso = self._riso_step(g1, g2)
        w, wmi = self._warmups()

        # Terms summed over genes are divided by the number of genes; aux and riso are not.
        per_gene = recon + w * (o.beta_nr * kl_nr + o.beta_r * kl_r) + wmi * o.lambda_club_u * club
        loss = per_gene / x.shape[1] + wmi * (o.lambda_aux * aux + o.lambda_riso * riso)

        self.log_dict({"loss": loss, "recon": recon, "kl_nr": kl_nr, "kl_r": kl_r,
                       "club": club, "aux": aux, "riso": riso})
        return loss

    def _club_step(self, z_nr, g1, g2):
        """Fit the CLUB critic (`club_inner` steps, its own optimiser), then estimate MI(z_nr; e_u)."""
        u = self.pert_enc(g1, g2).detach()
        z = z_nr.detach()
        for _ in range(self.cfg.objective.club_inner):
            self.club_opt.zero_grad(set_to_none=True)
            self.club_u.loglikeli(z, u).backward()
            self.club_opt.step()
        return self.club_u.mi_est(z_nr, u).clamp_min(0.0)

    def _build_aux(self):
        """Mean response (expr - control mean) of each training condition, keyed by (g1, g2)."""
        d = self.data
        lab = np.asarray([str(c) for c in d.train.condition])
        X = d.train.expr.float()
        ctrl = X[torch.as_tensor(lab == "control", device=X.device)].mean(0)
        n = self.pert_enc.n_genes
        keys, rows = [], []
        for c in sorted(set(lab) - {"control"}):
            i = int(np.flatnonzero(lab == c)[0])
            keys.append(int(d.train.g1[i]) * (n + 1) + int(d.train.g2[i]))
            rows.append(X[torch.as_tensor(lab == c, device=X.device)].mean(0) - ctrl)
        k = torch.tensor(keys, device=X.device)
        order = k.argsort()
        self.register_buffer("_aux_key", k[order], persistent=False)
        self.register_buffer("_aux_val", torch.stack(rows)[order], persistent=False)

    def _lookup(self, g1, g2):
        """Rows of the training mean responses for (g1, g2); `ok` marks conditions that have one."""
        if not hasattr(self, "_aux_key"):
            self._build_aux()
        key = g1 * (self.pert_enc.n_genes + 1) + g2
        pos = torch.searchsorted(self._aux_key, key).clamp_max(len(self._aux_key) - 1)
        return pos, self._aux_key[pos] == key

    def _aux_step(self, g1, g2):
        """Response supervision: a linear read-out of e_u predicts the condition's mean response."""
        if self.cfg.objective.lambda_aux <= 0:
            return self.pert_enc.null.new_zeros(())
        n = self.pert_enc.n_genes
        keep = (g1 != n) | (g2 != n)
        if int(keep.sum()) == 0:
            return self.pert_enc.null.new_zeros(())
        g1, g2 = g1[keep], g2[keep]
        pos, ok = self._lookup(g1, g2)
        if int(ok.sum()) == 0:
            return self.pert_enc.null.new_zeros(())
        pred = self.aux_head(self.pert_enc(g1[ok], g2[ok]))
        return ((pred - self._aux_val[pos[ok]]) ** 2).mean()

    def _riso_step(self, g1, g2):
        """Relative isometry: e_u distances follow response distances over the batch's conditions."""
        if self.cfg.objective.lambda_riso <= 0:
            return self.pert_enc.null.new_zeros(())
        n = self.pert_enc.n_genes
        keep = (g1 != n) | (g2 != n)
        if int(keep.sum()) < 3:
            return self.pert_enc.null.new_zeros(())
        uniq = torch.unique(torch.stack([g1[keep], g2[keep]], 1), dim=0)
        if uniq.shape[0] < 3:
            return self.pert_enc.null.new_zeros(())
        a, b = uniq[:, 0], uniq[:, 1]
        pos, ok = self._lookup(a, b)
        if int(ok.sum()) < 3:
            return self.pert_enc.null.new_zeros(())
        return isometry_loss(self._aux_val[pos[ok]], self.pert_enc(a[ok], b[ok]))

    def configure_optimizers(self):
        """Adam over everything except the CLUB critic."""
        club_ids = {id(p) for p in self.club_u.parameters()}
        return torch.optim.Adam([p for p in self.parameters() if id(p) not in club_ids],
                                lr=self.cfg.train.lr)

    def _control_block(self):
        """(decoded, real) training control cells: the reference every response is measured from."""
        d = self.data
        ctrl = torch.nonzero(d.train.is_control, as_tuple=True)[0]
        s = self.encode_all(d.train.counts[ctrl].float(), c=d.train.covar[ctrl])
        pred = self.dec.to_eval_space(self.dec(s["z_nr"], s["z_r"])).cpu().numpy()
        return pred, d.train.expr[ctrl].float().cpu().numpy()
