"""Evaluation metrics, following the definitions of scBIG.

    rho_delta    Pearson correlation of predicted and true shifts from control
    rho_delta_D  the same on the 20 genes with the largest true shift
    ACC_delta    fraction of genes whose direction of change is right
    DES          Spearman correlation of log fold changes over significant DE genes
    PDS_sb       rank of the true condition among all candidates, by distance to the prediction

All are computed on pseudobulk means, except `scbig_distributional`, which compares cell populations.
"""
from __future__ import annotations

import re

import numpy as np


def _parse_cond_genes(cond: str) -> list[str]:
    return [g for g in re.split(r"[+_]", str(cond)) if g not in ("ctrl", "control", "")]


def _n_genes_in(cond: str) -> int:
    return len(_parse_cond_genes(cond))

def pseudobulk(x: np.ndarray, labels) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    return {c: x[labels == c].mean(0) for c in np.unique(labels)}


def compute_metrics(pred: np.ndarray, real: np.ndarray, pred_labels, real_labels,
                    control: str = "control", top_de: int = 20) -> dict:
    """Metrics averaged over the non-control conditions present in both pred and real."""
    pb_pred, pb_real = pseudobulk(pred, pred_labels), pseudobulk(real, real_labels)
    conds = [c for c in pb_real if c != control and c in pb_pred]
    keys = ("rho_delta", "rho_delta_D", "ACC_delta", "ACC_delta_D", "L2", "MSE", "MAE",
            "DES")
    if not conds or control not in pb_real:
        return {k: float("nan") for k in keys}

    ctrl_r = pb_real[control]
    rho, rhoD, acc, accD, l2, mse, mae = [], [], [], [], [], [], []

    for i, c in enumerate(conds):
        dp = pb_pred[c] - ctrl_r
        dr = pb_real[c] - ctrl_r
        if dp.std() > 1e-9 and dr.std() > 1e-9:
            rho.append(np.corrcoef(dp, dr)[0, 1])
        top = np.argsort(np.abs(dr))[-top_de:]
        if dp[top].std() > 1e-9:
            rhoD.append(np.corrcoef(dp[top], dr[top])[0, 1])
        acc.append(np.mean(np.sign(dp) == np.sign(dr)))
        accD.append(np.mean(np.sign(dp[top]) == np.sign(dr[top])))
        l2.append(np.linalg.norm(pb_pred[c] - pb_real[c]))
        mse.append(np.mean((pb_pred[c] - pb_real[c]) ** 2))
        mae.append(np.mean(np.abs(pb_pred[c] - pb_real[c])))

    return dict(rho_delta=float(np.nanmean(rho)), rho_delta_D=float(np.nanmean(rhoD)),
                ACC_delta=float(np.mean(acc)), ACC_delta_D=float(np.mean(accD)),
                DES=des_scbig(pred, real, pred_labels, real_labels, control=control),
                L2=float(np.mean(l2)),
                MSE=float(np.mean(mse)), MAE=float(np.mean(mae)))

def pds_scbig(pb_pred, pb_real, conds, genes) -> dict[str, float]:
    """Perturbation discrimination score per condition: 1 - (rank - 1) / (N - 1).

    The rank is that of the true condition among all candidates by L1 distance to the prediction,
    with the perturbed genes left out.
    """
    pool = list(conds)
    idx = {c: i for i, c in enumerate(pool)}
    true_mat = np.stack([pb_real[c] for c in pool])
    gidx = {g: i for i, g in enumerate(genes)}
    out = {}
    for c in conds:
        keep = np.ones(true_mat.shape[1], dtype=bool)
        for g in re.split(r"[+_]", str(c)):
            j = gidx.get(g)
            if j is not None:
                keep[j] = False
        d = np.abs(true_mat[:, keep] - pb_pred[c][keep][None, :]).sum(1)
        rank = int(np.argsort(np.argsort(d))[idx[c]]) + 1
        out[c] = 1.0 - (rank - 1) / max(len(pool) - 1, 1)
    return out


def metrics_by_arity(pred, real, pred_labels, real_labels, control: str = "control",
                     dataset: str = "holdout", genes=None) -> dict:
    """Metrics per arity: 'single' and 'double' rows, one 'additive' row on the additive split and
    one 'pooled' row on ComboSciPlex. PDS_sb uses one candidate pool over all conditions.
    """
    additive = dataset == "additive"
    out = {}
    if additive:
        out["additive"] = compute_metrics(pred, real, pred_labels, real_labels, control=control)
    if dataset == "combosciplex":
        out["pooled"] = compute_metrics(pred, real, pred_labels, real_labels, control=control)
    for nd, tag in () if additive else ((1, "single"), (2, "double")):
        kp = [i for i, l in enumerate(pred_labels) if l == control or _n_genes_in(l) == nd]
        kr = [i for i, l in enumerate(real_labels) if l == control or _n_genes_in(l) == nd]
        if not any(pred_labels[i] != control for i in kp) or \
           not any(real_labels[i] != control for i in kr):
            continue
        out[tag] = compute_metrics(pred[kp], real[kr],
                                   [pred_labels[i] for i in kp], [real_labels[i] for i in kr],
                                   control=control)

    pb_pred, pb_real = pseudobulk(pred, pred_labels), pseudobulk(real, real_labels)
    pool = [c for c in pb_real if c != control and c in pb_pred]
    if pool:
        per_cond = pds_scbig(pb_pred, pb_real, pool, genes=genes)
        for nd, tags in ((1, ("single",)), (2, ("double", "additive"))):
            v = [s for c, s in per_cond.items() if _n_genes_in(c) == nd]
            if not v:
                continue
            for tag in tags:
                if tag in out:
                    out[tag]["PDS_sb"] = float(np.mean(v))
        if "pooled" in out and per_cond:
            out["pooled"]["PDS_sb"] = float(np.mean(list(per_cond.values())))

    return out


def des_scbig(pred_mat, real_mat, pred_labels, real_labels, control="control", p_thresh=0.05):
    """Spearman correlation of predicted and true log fold changes over significant genes.

    Significance: Wilcoxon rank-sum test, Bonferroni-corrected.
    """
    from scipy.stats import ranksums, spearmanr
    eps = 1e-10
    rl = np.asarray(real_labels); pl = np.asarray(pred_labels)
    ctrl_cells = real_mat[rl == control]
    if not len(ctrl_cells):
        return float("nan")
    ctrl_mean = ctrl_cells.mean(0)
    n_genes = real_mat.shape[1]
    out = []
    for c in [x for x in np.unique(rl) if x != control]:
        cond_cells = real_mat[rl == c]
        pred_cells = pred_mat[pl == c]
        if not len(cond_cells) or not len(pred_cells):
            continue
        p = np.ones(n_genes)
        for g in range(n_genes):
            a, b = cond_cells[:, g], ctrl_cells[:, g]
            if a.std() == 0 and b.std() == 0:
                continue
            p[g] = ranksums(a, b).pvalue
        sig = np.flatnonzero(np.minimum(p * n_genes, 1.0) < p_thresh)
        if len(sig) < 5:
            continue
        t = np.log2((cond_cells.mean(0)[sig] + eps) / (ctrl_mean[sig] + eps))
        q = np.log2((pred_cells.mean(0)[sig] + eps) / (ctrl_mean[sig] + eps))
        ok = np.isfinite(t) & np.isfinite(q)
        if ok.sum() < 5:
            continue
        r = spearmanr(q[ok], t[ok]).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out)) if out else 0.0


def e_distance(x: np.ndarray, y: np.ndarray, max_n: int = 600) -> float:
    """Energy distance 2E|x-y| - E|x-x'| - E|y-y'| on the first `max_n` cells of each population."""
    x, y = x[:max_n], y[:max_n]

    def d(a, b):
        m = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
        np.maximum(m, 0, out=m)
        return np.sqrt(m).mean()

    return float(2 * d(x, y) - d(x, x) - d(y, y))


def _pca_project(fit_on: np.ndarray, *sets: np.ndarray, n_comp: int = 50, seed: int = 0):
    """Fit a PCA on `fit_on` and project every set into it."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(n_comp, fit_on.shape[1], max(2, fit_on.shape[0] - 1)),
            random_state=seed).fit(fit_on)
    return [p.transform(s) for s in sets]


def scbig_distributional(pred: np.ndarray, real: np.ndarray, pred_labels, real_labels,
                         control: str = "control", n_comp: int = 50, eps: float = 1.0,
                         max_n: int = 600, seed: int = 0, max_cond: int = 0) -> dict:
    """Distribution-level metrics of scBIG, averaged over perturbations.

    discr_cos    cosine between the true and predicted mean shifts from control
    e_dist       energy distance, in a 50-dim PCA space fitted on real cells
    wasserstein  entropic OT cost, in the same space
    """
    import ot as pot
    pred_labels = np.asarray(pred_labels)
    real_labels = np.asarray(real_labels)
    conds = [c for c in sorted(set(real_labels) & set(pred_labels)) if c != control]
    if not conds or (real_labels == control).sum() == 0:
        return {}
    # optional fixed subsample of the conditions
    if max_cond and len(conds) > max_cond:
        conds = sorted(np.random.RandomState(seed).choice(conds, max_cond, replace=False).tolist())

    Zr_all, Zp_all = _pca_project(real, real, pred, n_comp=n_comp, seed=seed)
    gctrl_mu = real[real_labels == control].mean(0)

    cos, ed, wass = [], [], []
    rng = np.random.default_rng(seed)
    for c in conds:
        ri = np.flatnonzero(real_labels == c)
        pi = np.flatnonzero(pred_labels == c)
        if len(ri) < 2 or len(pi) < 2:
            continue
        dr, dp = real[ri].mean(0) - gctrl_mu, pred[pi].mean(0) - gctrl_mu
        nr, np_ = np.linalg.norm(dr), np.linalg.norm(dp)
        if nr > 0 and np_ > 0:
            cos.append(float(dr @ dp / (nr * np_)))

        A = Zr_all[ri if len(ri) <= max_n else rng.choice(ri, max_n, replace=False)]
        B = Zp_all[pi if len(pi) <= max_n else rng.choice(pi, max_n, replace=False)]
        ed.append(e_distance(A, B, max_n=max_n))
        C = np.sqrt(np.maximum(
            (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T), 0))
        a = np.ones(len(A)) / len(A)
        b = np.ones(len(B)) / len(B)
        wass.append(float(pot.sinkhorn2(a, b, np.ascontiguousarray(C, dtype=np.float64), eps)))

    out = {}
    if cos:
        out["discr_cos"] = float(np.mean(cos))
    if ed:
        out["e_dist"] = float(np.mean(ed))
    if wass:
        out["wasserstein"] = float(np.mean(wass))
    out["n_cond_dist"] = len(ed)
    return out

