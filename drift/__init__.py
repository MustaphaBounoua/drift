"""DRIFT: disentangled responsive-invariant flow transport for single-cell perturbation prediction.

    stage 1   a disentangled VAE splits each cell into an invariant block z_nr and a responsive
              block z_r (drift/vae.py);
    stage 2   a conditional flow transports z_r from a real control cell to the perturbed state
              (drift/flow.py).
"""
from .data import PerturbData
from .features import esm2_features
from .metrics import compute_metrics, metrics_by_arity

__all__ = ["PerturbData", "esm2_features", "compute_metrics", "metrics_by_arity"]
__version__ = "1.0.0"
