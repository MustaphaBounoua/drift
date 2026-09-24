"""Prior features from STRING interactions and Reactome pathway membership.

Each is a binary gene-by-partner membership matrix reduced to `dim` components. Genes with no
annotation keep an all-zero row rather than the shared centering direction.

Inputs are the files of scBIG's prior-resources release, placed in `resources/`:
    9606.protein.info.v12.0.txt.gz, 9606.protein.links.v12.0.txt.gz   STRING v12.0, human
    ReactomePathways.gmt                                                 Reactome pathways
The reduced tables are generated on first use and cached in `resources/cache/`.
"""
from __future__ import annotations

import gzip
import pathlib

import torch


RESOURCES = pathlib.Path("resources")
STRING_INFO = RESOURCES / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS = RESOURCES / "9606.protein.links.v12.0.txt.gz"
REACTOME_GMT = RESOURCES / "ReactomePathways.gmt"
CACHE = RESOURCES / "cache"


def load_string(min_score: int = 400, verbose: bool = True):
    """STRING edges, as gene-symbol pairs, with combined score >= `min_score`."""
    prot2sym = {}
    with gzip.open(STRING_INFO, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                prot2sym[p[0]] = p[1]
    edges = []
    with gzip.open(STRING_LINKS, "rt") as fh:
        next(fh)
        for line in fh:
            a, b, score = line.split()
            if int(score) < min_score:
                continue
            ga, gb = prot2sym.get(a), prot2sym.get(b)
            if ga and gb and ga != gb:
                edges.append((ga, gb))
    return edges


_FEATURE_SVD_SEED = 0


def _seeded_pca(X, q):
    """`torch.pca_lowrank` under a fixed seed, leaving the global RNG state untouched."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(_FEATURE_SVD_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_FEATURE_SVD_SEED)
        return torch.pca_lowrank(X, q=q)
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def ppi_matrix(gene_names, edges, dim: int = 128, device: str = "cuda", verbose: bool = True):
    """Row-normalised STRING adjacency of each panel gene, reduced to `dim` by PCA.

    Genes without an interactor get an all-zero row.
    """
    universe = sorted({g for e in edges for g in e})
    uidx = {g: i for i, g in enumerate(universe)}
    idx = {g: i for i, g in enumerate(gene_names)}
    n, m = len(gene_names), len(universe)
    A = torch.zeros(n, m, device=device)
    hit = 0
    for a, b in edges:
        ia, ib = idx.get(a), idx.get(b)
        ja, jb = uidx.get(a), uidx.get(b)
        if ia is not None and jb is not None:
            A[ia, jb] = 1.0; hit += 1
        if ib is not None and ja is not None:
            A[ib, ja] = 1.0
    if verbose:
        deg = (A.sum(1) > 0).float().mean()
        print(f"[string] {hit:,} panel-to-universe edges over {m:,} partner coordinates; "
              f"{100*float(deg):.0f}% of panel genes have >=1 interactor", flush=True)
    M = A
    seen = M.sum(1) > 0
    M = torch.nn.functional.normalize(M, dim=1)
    q = min(dim, min(M.shape) - 1)
    _, _, V = _seeded_pca(M - M.mean(0, keepdim=True), q)
    Z = torch.nn.functional.normalize((M - M.mean(0, keepdim=True)) @ V, dim=1)
    return Z * seen.unsqueeze(1).to(Z.dtype)


def _cached(kind: str, key: str, build, device: str, verbose: bool):
    import hashlib
    f = CACHE / f"{kind}_{hashlib.md5(key.encode()).hexdigest()[:16]}.pt"
    if f.exists():
        if verbose:
            print(f"[{kind}] cache hit {f.name}", flush=True)
        return torch.load(f, map_location=device)
    Z = build()
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Z.cpu(), f)
    if verbose:
        print(f"[{kind}] generated -> {f}", flush=True)
    return Z.to(device)


def cached_ppi(gene_names, dim: int = 128, min_score: int = 400, device: str = "cuda",
               verbose: bool = True):
    """`ppi_matrix` for a gene panel, generated from STRING once and cached."""
    key = "|".join(gene_names) + f"|{dim}|{min_score}"
    return _cached("string", key, lambda: ppi_matrix(
        gene_names, load_string(min_score, verbose), dim, device, verbose), device, verbose)


def pathway_matrix(gene_names, g2p, dim: int = 128, device: str = "cuda", verbose: bool = True):
    """Pathway membership of each panel gene, reduced to `dim` by PCA; unannotated genes get zero rows."""
    paths = sorted({p for g in gene_names for p in g2p.get(g, ())})
    if not paths:
        return torch.zeros(len(gene_names), dim, device=device)
    pidx = {p: i for i, p in enumerate(paths)}
    M = torch.zeros(len(gene_names), len(paths), device=device)
    hit = 0
    for i, g in enumerate(gene_names):
        ps = g2p.get(g, ())
        if ps:
            hit += 1
        for p in ps:
            M[i, pidx[p]] = 1.0
    if verbose:
        print(f"[reactome] {hit}/{len(gene_names)} genes annotated over {len(paths):,} pathways",
              flush=True)
    seen = M.sum(1) > 0
    M = torch.nn.functional.normalize(M, dim=1)
    q = min(dim, min(M.shape) - 1)
    _, _, V = _seeded_pca(M - M.mean(0, keepdim=True), q)
    Z = torch.nn.functional.normalize((M - M.mean(0, keepdim=True)) @ V, dim=1)
    return Z * seen.unsqueeze(1).to(Z.dtype)


def cached_pathway(gene_names, dim: int = 128, device: str = "cuda", verbose: bool = True):
    """`pathway_matrix` for a gene panel, generated from the Reactome GMT once and cached."""
    key = "|".join(gene_names) + f"|{dim}|gmt"
    return _cached("reactome", key, lambda: pathway_matrix(
        gene_names, load_reactome_gmt(REACTOME_GMT, verbose), dim, device, verbose),
        device, verbose)


def load_reactome_gmt(path=REACTOME_GMT, verbose: bool = True):
    """{gene_symbol: set(pathway_ids)} from ReactomePathways.gmt."""
    import collections
    g2p = collections.defaultdict(set)
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            pid = p[1] or p[0]
            for g in p[2:]:
                if g:
                    g2p[g].add(pid)
    if verbose:
        print(f"[reactome-gmt] {len(g2p):,} gene symbols over "
              f"{len({p for s in g2p.values() for p in s}):,} pathways", flush=True)
    return g2p


