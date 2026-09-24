"""Evaluation on the test split.

    score_genetic   Norman and Replogle: scBIG's metrics per arity and per seen/unseen group
    score_celleval  ComboSciPlex: cell-eval, following scDFM
"""
from __future__ import annotations

import csv
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .data import gene_groups
from .metrics import metrics_by_arity, scbig_distributional

KEYS = ("rho_delta", "rho_delta_D", "ACC_delta", "ACC_delta_D", "DES", "PDS_sb", "L2", "MSE", "MAE")
DIST_KEYS = ("discr_cos", "e_dist", "wasserstein")
EVAL_SEED = 1000


@torch.no_grad()
def generate_split(flow, split: str):
    """Generate `n_gen` cells for every condition of `split`, plus the control reference block.

    Returns (pred, real, pred_labels, real_labels) in eval space.
    """
    sp = getattr(flow.data, split)
    lab = np.asarray([str(c) for c in sp.condition])
    conds = [c for c in sorted(set(lab)) if c != "control"]
    P, pl = [], []
    for c in conds:
        i = int(np.flatnonzero(lab == c)[0])
        P.append(flow.generate(int(sp.g1[i]), int(sp.g2[i]), n=flow.cfg.train.n_gen))
        pl += [c] * len(P[-1])
    # controls come from `_control_block`; the split's own control cells are left out
    cp, cr = flow.vae._control_block()
    keep = lab != "control"
    pred = np.concatenate([cp] + P)
    real = np.concatenate([cr, sp.expr.float().cpu().numpy()[keep]])
    return (pred, real,
            np.asarray(["control"] * len(cp) + pl),
            np.asarray(["control"] * len(cr) + list(lab[keep])))


def score_genetic(flow, n_gen: int = 500, repeats: int = 1, dist_max_cond: int = 15) -> dict:
    """Returns {"<arity>/<group>": {metric: value}, "dist": {...}}.

    Each metric is the mean over `repeats` generation passes, seeded EVAL_SEED + r. Groups are all
    conditions, and the conditions whose genes were all seen (or not all seen) in training. The
    distribution-level metrics use `dist_max_cond` conditions.
    """
    data = flow.data
    flow.cfg.train.n_gen = n_gen
    acc: dict[str, dict[str, list[float]]] = {}
    for r in range(repeats):
        torch.manual_seed(EVAL_SEED + r)
        pred, real, pl, rl = generate_split(flow, "test")
        cn = int((rl == "control").sum())
        for gname, mask in [("all", None)] + sorted(gene_groups(data, "test").items()):
            if mask is None:
                P, R, p_, r_ = pred, real, pl, rl
            else:
                keep = {c for c in np.unique(rl[cn:][mask]) if c != "control"}
                pm = np.array([c == "control" or c in keep for c in pl])
                rm = np.concatenate([np.ones(cn, bool), mask])
                P, R, p_, r_ = pred[pm], real[rm], pl[pm], rl[rm]
            if len(set(r_)) < 2:
                continue
            for arity, m in metrics_by_arity(P, R, p_, r_, genes=data.genes,
                                             dataset=data.dataset).items():
                for k in KEYS:
                    if k in m:
                        acc.setdefault(f"{arity}/{gname}", {}).setdefault(k, []).append(float(m[k]))
            if gname == "all":
                dm = scbig_distributional(P, R, p_, r_, max_cond=dist_max_cond)
                for k in DIST_KEYS:
                    if k in dm:
                        acc.setdefault("dist", {}).setdefault(k, []).append(float(dm[k]))
    return {g: {k: float(np.mean(v)) for k, v in m.items()} for g, m in acc.items()}


# cell-eval key -> reported name; L2 is computed here.
CELLEVAL_COLUMNS = {"pearson_delta": "rho_delta", "de_spearman_lfc_sig": "DE_spearman",
                    "discrimination_score_l1": "DS", "mse": "MSE", "mae": "MAE"}


def _write_h5ad(path, X, labels, genes, col):
    import anndata as ad
    a = ad.AnnData(X=np.asarray(X, dtype=np.float32))
    a.obs[col] = [str(x) for x in labels]
    a.obs[col] = a.obs[col].astype("category")
    a.var_names = [str(g) for g in genes]
    a.write_h5ad(path)


def score_celleval(flow, n_gen: int = 128) -> dict:
    """ComboSciPlex under scDFM's protocol: 128 cells per condition on the 7 test conditions,
    scored on the 1000 test-selected HVGs by cell-eval.
    """
    data = flow.data
    flow.cfg.train.n_gen = n_gen
    torch.manual_seed(EVAL_SEED)
    pred, real, pl, rl = generate_split(flow, "test")
    eg = data.eval_genes.cpu().numpy()
    pred, real = pred[:, eg], real[:, eg]
    genes = [str(g) for g in np.asarray(data.genes)[eg]]

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="drift_celleval_"))
    try:
        _write_h5ad(tmp / "pred.h5ad", pred, pl, genes, "target_gene")
        _write_h5ad(tmp / "real.h5ad", real, rl, genes, "target_gene")
        cmd = [sys.executable, "-m", "cell_eval", "run", "-ap", str(tmp / "pred.h5ad"), "-ar", str(tmp / "real.h5ad"),
               "--control-pert", "control", "--pert-col", "target_gene", "-o", str(tmp / "out"),
               "--num-threads", "8"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cell-eval exited {r.returncode}:\n{r.stderr[-2000:]}")
        rows = list(csv.DictReader(open(tmp / "out" / "agg_results.csv")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    mean = next(row for row in rows if row["statistic"] == "mean")
    out = {name: float(mean[k]) for k, name in CELLEVAL_COLUMNS.items()}

    pl, rl = np.asarray([str(x) for x in pl]), np.asarray([str(x) for x in rl])
    conds = sorted({c for c in rl if c != "control"} & {c for c in pl if c != "control"})
    out["L2"] = float(np.mean([np.linalg.norm(pred[pl == c].mean(0) - real[rl == c].mean(0))
                               for c in conds]))
    return out
