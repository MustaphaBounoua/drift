"""Fixed perturbation features, one row per perturbed gene or drug.

    esm2_features      ESM2 protein embeddings, z-scored and PCA-reduced (genetic perturbations)
    morgan_features    Morgan fingerprints, standardised and PCA-reduced (drugs)
"""
from __future__ import annotations

import os
import pickle

import torch


def esm2_features(pert_gene_vocab: dict[str, int], n_pert_genes: int, esm2_path: str,
                  d_feat: int = 128, verbose: bool = True) -> torch.Tensor:
    """(n_pert_genes, d_feat) ESM2 table in the row order of `pert_gene_vocab`; missing genes are zero."""
    if not os.path.exists(esm2_path):
        raise FileNotFoundError(
            f"ESM2 features not found at {esm2_path}. See the Data section of the README.")
    esm = torch.load(esm2_path, map_location="cpu")           # {gene: (5120,) tensor}
    names = list(esm.keys())
    vecs = torch.stack([esm[g] for g in names]).float()
    z = (vecs - vecs.mean(0)) / vecs.std(0).clamp_min(1e-6)
    _, _, V = torch.pca_lowrank(z, q=min(d_feat, z.shape[1]), center=True, niter=4)
    proj = z @ V
    table = {g: proj[i] for i, g in enumerate(names)}
    W = torch.zeros(n_pert_genes, proj.shape[1])
    missing = []
    for g, gid in pert_gene_vocab.items():
        if g in table:
            W[gid] = table[g]
        else:
            missing.append(g)
    if verbose and missing:
        print(f"[features] genes without ESM2 features: {missing}", flush=True)
    return W


def morgan_features(pert_vocab: dict[str, int], n_pert: int, path: str = "resources/combo_morgan_fp.pkl",
                    d_feat: int = 16) -> torch.Tensor:
    """Morgan fingerprints (radius 4, 1024 bits) of the ComboSciPlex drugs."""
    fp = pickle.load(open(path, "rb"))
    W = torch.zeros(n_pert, len(next(iter(fp.values()))))
    for g, i in pert_vocab.items():
        if g in fp:
            W[i] = torch.as_tensor(fp[g])
    W = (W - W.mean(0)) / W.std(0).clamp_min(1e-6)
    _, _, V = torch.pca_lowrank(W, q=min(d_feat, min(W.shape) - 1), center=True, niter=4)
    W = W @ V
    return W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)


def feature_table(data, dcfg, device="cpu", verbose=False) -> torch.Tensor:
    """The perturbation feature table for a DataConfig."""
    if dcfg.features == "morgan":
        return morgan_features(data.pert_gene_vocab, data.n_pert_genes, d_feat=dcfg.d_feat).to(device)
    return esm2_features(data.pert_gene_vocab, data.n_pert_genes, dcfg.esm2_path, dcfg.d_feat,
                         verbose=verbose).to(device)
