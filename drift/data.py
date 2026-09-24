"""Datasets on their published splits, as GPU tensors.

Conditions are normalised to 'control' or 'geneA+geneB'.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def _parse_genes(c: str) -> list[str]:
    """'A+B' | 'A_B' | 'A_ctrl' | 'ctrl' -> list of perturbed genes."""
    return [g for g in re.split(r"[+_]", str(c)) if g not in ("ctrl", "control", "")]


@dataclass
class Split:
    """One split; rows are cells."""

    counts: torch.Tensor      # (N, G) counts, encoder input
    expr: torch.Tensor        # (N, G) log-normalised expression, decoder target
    g1: torch.Tensor          # (N,)   first perturbed gene id; n_pert_genes means none
    g2: torch.Tensor          # (N,)   second perturbed gene id
    is_control: torch.Tensor  # (N,)
    condition: np.ndarray     # (N,)   'control' | 'geneA+geneB'
    covar: torch.Tensor       # (N, C) covariates c


# Covariates c: (categorical, numeric) columns of each dataset. ComboSciPlex has none: in an
# arrayed drug screen the well, and the cell-level QC measures, depend on the drug.
COVARIATES = {
    "norman": (("gemgroup",), ("ncounts", "ngenes", "percent_mito", "percent_ribo")),
    "replogle": (("gem_group",), ("mitopercent", "UMI_count", "n_genes")),
    "combosciplex": ((), ()),
}


class PerturbData:
    """Norman (additive, holdout), Replogle RPE1 or ComboSciPlex, as train and test splits."""

    def __init__(self, data_dir: str, dataset: str = "holdout", device: str = "cuda"):
        assert dataset in ("holdout", "additive", "combosciplex", "replogle")
        if dataset == "replogle":
            atr, ava, ate = self._read_replogle(data_dir)
        elif dataset == "combosciplex":
            atr, ava, ate = self._read_combosciplex(data_dir)
        else:
            pfx = "norman_holdout_" if dataset == "holdout" else "norman_"
            atr = ad.read_h5ad(f"{data_dir}/{pfx}train.h5ad")
            ate = ad.read_h5ad(f"{data_dir}/{pfx}test.h5ad")
            ava = ad.read_h5ad(f"{data_dir}/{pfx}val.h5ad")
        self.device, self.dataset = device, dataset
        self._covar_cat, self._covar_num = COVARIATES.get(dataset, COVARIATES["norman"])
        self.genes = list(atr.var_names)

        # Perturbation vocabulary over all splits, so held-out genes have feature rows too.
        allc = [str(c) for a in (atr, ate, ava) for c in a.obs["condition"]]
        vocab = sorted({g for c in allc for g in _parse_genes(c)})
        self.pert_gene_vocab = {g: i for i, g in enumerate(vocab)}
        self.n_pert_genes = len(vocab)          # id n_pert_genes means no gene

        self._covar_levels: dict[str, list[str]] = {}
        self._covar_stats: dict[str, tuple[float, float]] = {}
        self.train = self._build(atr)
        self.test = self._build(ate)
        self.n_covar = self.train.covar.shape[1]

        # ComboSciPlex is scored on 1000 HVGs selected on the test set, as in scDFM.
        self.eval_genes = None
        if dataset == "combosciplex":
            import scanpy as sc
            sub = ad.AnnData(X=_dense(ate.X).astype(np.float32))
            sc.pp.highly_variable_genes(sub, n_top_genes=1000)
            self.eval_genes = torch.as_tensor(
                np.flatnonzero(sub.var["highly_variable"].values), device=device)

    @staticmethod
    def _read_replogle(data_dir: str):
        """Replogle RPE1 (CRISPRi) from scBIG's released split files."""
        paths = {s: os.path.join(data_dir, "rpe1", f"RPE1_{s}_filtered.h5ad")
                 for s in ("train", "val", "test")}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Replogle RPE1 split not found: " + ", ".join(missing) +
                                    ". See the Data section of the README.")
        return tuple(ad.read_h5ad(paths[s]) for s in ("train", "val", "test"))

    @staticmethod
    def _read_combosciplex(data_dir: str):
        """ComboSciPlex (A549, drug combinations) with scDFM's preprocessing and test conditions."""
        import scanpy as sc
        a = ad.read_h5ad(os.path.join(data_dir, "combosciplex.h5ad"))
        a.X = a.layers["counts"].copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=5000)
        a = a[:, a.var["highly_variable"].values].copy()

        cond = []
        for x, y in zip(a.obs["Drug1"].astype(str).values, a.obs["Drug2"].astype(str).values):
            gs = [g for g in (x, y) if g not in ("control", "ctrl", "nan", "")]
            cond.append("control" if not gs else "+".join(gs))
        a.obs["condition"] = np.asarray(cond, dtype=object)
        a.obs["is_control"] = np.asarray([c == "control" for c in cond])

        test = ["Panobinostat+Crizotinib", "Panobinostat+Curcumin", "Panobinostat+SRT1720",
                "Panobinostat+Sorafenib", "SRT2104+Alvespimycin", "Alvespimycin", "Dacinostat"]
        ist = np.isin(a.obs["condition"].astype(str).values, test)
        # Control cells are in both train and test, as in scDFM; there is no separate val.
        ate = a[ist | a.obs["is_control"].values].copy()
        return a[~ist].copy(), ate, ate

    def _covariates(self, adata) -> np.ndarray:
        obs, parts = adata.obs, []
        for c in self._covar_cat:
            if c not in obs:
                continue
            codes = obs[c].astype(str).values
            levels = self._covar_levels.setdefault(c, sorted(set(codes)))
            oh = np.zeros((len(codes), len(levels)), dtype=np.float32)
            idx = {l: i for i, l in enumerate(levels)}
            for i, v in enumerate(codes):
                if v in idx:
                    oh[i, idx[v]] = 1.0
            parts.append(oh)
        for c in self._covar_num:
            if c not in obs:
                continue
            v = obs[c].astype(np.float32).values
            if c in ("ncounts", "ngenes"):
                v = np.log1p(v)
            # standardised with training statistics
            mu, sd = self._covar_stats.setdefault(c, (float(v.mean()), float(v.std()) or 1.0))
            parts.append(((v - mu) / sd).astype(np.float32)[:, None])
        return np.concatenate(parts, 1) if parts else np.zeros((adata.n_obs, 0), dtype=np.float32)

    def _build(self, adata) -> Split:
        expr = _dense(adata.X).astype(np.float32)
        if "counts" in adata.layers:
            counts = _dense(adata.layers["counts"]).astype(np.float32)
        else:
            counts = np.expm1(expr).astype(np.float32)          # files without raw counts
        cond = adata.obs["condition"].astype(str).values
        isc = (adata.obs["is_control"].astype(bool).values if "is_control" in adata.obs
               else np.array([len(_parse_genes(c)) == 0 for c in cond]))
        norm = np.array(["control" if isc[i] else "+".join(_parse_genes(cond[i]))
                         for i in range(len(cond))], dtype=object)
        g1, g2 = self._condition_genes(norm)
        d = self.device
        return Split(counts=torch.as_tensor(counts, device=d), expr=torch.as_tensor(expr, device=d),
                     g1=torch.as_tensor(g1, device=d), g2=torch.as_tensor(g2, device=d),
                     is_control=torch.as_tensor(isc, device=d), condition=norm.astype(str),
                     covar=torch.as_tensor(self._covariates(adata), device=d))

    def _condition_genes(self, condition):
        n = len(condition)
        g1 = np.full(n, self.n_pert_genes, dtype=np.int64)
        g2 = np.full(n, self.n_pert_genes, dtype=np.int64)
        for i, c in enumerate(condition):
            gs = _parse_genes(c)
            if len(gs) >= 1 and gs[0] in self.pert_gene_vocab:
                g1[i] = self.pert_gene_vocab[gs[0]]
            if len(gs) >= 2 and gs[1] in self.pert_gene_vocab:
                g2[i] = self.pert_gene_vocab[gs[1]]
        return g1, g2


def gene_groups(data, split: str) -> dict[str, np.ndarray]:
    """Row masks of a split: 'unseen' conditions contain a gene absent from every training condition."""
    tr = np.asarray([str(c) for c in data.train.condition])
    seen_genes = {g for c in set(tr) for g in _parse_genes(c)}
    lab = np.asarray([str(c) for c in getattr(data, split).condition])
    is_unseen = np.array([any(g not in seen_genes for g in _parse_genes(c)) for c in lab])
    return {"seen": ~is_unseen, "unseen": is_unseen}
