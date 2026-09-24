"""Hyperparameters. The dataclass defaults, together with DATASET_DEFAULTS, are the published recipe.

Training budgets are in epochs; `resolve` converts them to optimiser steps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class DataConfig:
    """Dataset and perturbation features."""
    data_dir: str = "data"
    dataset: str = "holdout"             # additive | holdout | replogle | combosciplex
    features: str = "esm2"               # esm2 | morgan
    d_feat: int = 128                    # PCA width of the ESM2 or fingerprint table
    pathway_feat: int = 128              # Reactome block width (0 = off)
    ppi_feat: int = 128                  # STRING block width (0 = off)
    ppi_min_score: int = 400
    esm2_path: str = "resources/ESM2_pert_features.pt"


@dataclass
class ArchConfig:
    """Network sizes."""
    d_nr: int = 64                       # invariant block
    d_r: int = 192                       # responsive block
    d_pert: int = 128                    # perturbation embedding e_u
    hidden: int = 1024
    pert_hidden: int = 256
    enc_covar: bool = True               # c is an encoder input
    prior_covar: bool = True             # p(z_nr|c); False = N(0, I)
    pert_noise: float = 0.0              # Gaussian noise on e_u during training


@dataclass
class ObjectiveConfig:
    """Loss weights and schedules; see `drift.vae` for the objective."""
    beta_nr: float = 4.0
    beta_r: float = 0.5
    lambda_club_u: float = 5.0           # invariance penalty CLUB(z_nr; e_u)
    lambda_aux: float = 10.0             # response supervision
    lambda_riso: float = 10.0            # relative isometry
    club_u_dim: int = 32
    club_inner: int = 5                  # critic steps per training step
    club_lr: float = 1e-3
    kl_warmup_epochs: float = 20.0
    mi_warmup_epochs: float = 20.0


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 0.0
    seed: int = 0
    ckpt_every_epochs: float = 120.0


def _build(kind, values: dict):
    names = {f.name for f in fields(kind)}
    return kind(**{k: v for k, v in (values or {}).items() if k in names})


def _from_cli(kinds: dict, a) -> dict:
    out = {}
    for group, kind in kinds.items():
        vals = {}
        for f in fields(kind):
            v = getattr(a, f.name, None)
            if v is not None:
                vals[f.name] = bool(v) if isinstance(f.default, bool) else v
        out[group] = kind(**vals)
    return out


@dataclass
class VAEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    KINDS = dict(data=DataConfig, arch=ArchConfig, objective=ObjectiveConfig, train=TrainConfig)

    def resolve(self, n_train: int) -> dict:
        spe = max(1, -(-n_train // self.train.batch_size))
        return dict(
            steps_per_epoch=spe,
            max_steps=int(round(self.train.epochs * spe)),
            kl_warmup_steps=int(round(self.objective.kl_warmup_epochs * spe)),
            mi_warmup_steps=int(round(self.objective.mi_warmup_epochs * spe)),
            ckpt_every=max(1, int(round(self.train.ckpt_every_epochs * spe))),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VAEConfig":
        data = dict(d.get("data") or {})
        if "split_method" in data:                              # key name in older checkpoints
            data["dataset"] = data.pop("split_method")
        d = {**d, "data": data}
        return cls(**{k: _build(kind, d.get(k)) for k, kind in cls.KINDS.items()})

    @classmethod
    def from_cli(cls, a) -> "VAEConfig":
        return cls(**_from_cli(cls.KINDS, a))

    def describe(self) -> str:
        return "\n".join(f"[{g}] " + "  ".join(f"{k}={v}" for k, v in asdict(getattr(self, g)).items())
                         for g in self.KINDS)


@dataclass
class FlowArchConfig:
    """The DiT velocity field and its conditioning."""
    dit_layers: int = 6
    dit_tokens: int = 16
    cond_dim: int = 256
    hidden_dropout: float = 0.1
    cond_out_dropout: float = 0.0        # dropout on the condition during training
    pert_noise: float = 0.0              # Gaussian noise on the condition during training


@dataclass
class PairingConfig:
    """Entropic OT pairing of control and perturbed cells, re-solved every round."""
    n_cond: int = 16                     # conditions per round, drawn uniformly
    block: int = 512                     # cells per condition per round
    alpha: float = 1.0                   # weight of z_nr in the OT cost
    lam: float = 1.0                     # weight of z_r in the OT cost
    ot_reg: float = 0.5


@dataclass
class FlowTrainConfig:
    epochs: int = 3750                   # OT rounds; 3750 x 16 steps = 60k steps
    flow_batch: int = 512
    lr: float = 1e-4
    ema_decay: float = 0.999             # generation uses the EMA weights
    grad_clip: float = 0.0
    seed: int = 0
    ckpt_every_epochs: float = 1250.0    # in OT rounds
    ode_steps: int = 50                  # Euler steps at generation
    n_gen: int = 500                     # cells generated per condition


@dataclass
class FlowConfig:
    arch: FlowArchConfig = field(default_factory=FlowArchConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    train: FlowTrainConfig = field(default_factory=FlowTrainConfig)

    KINDS = dict(arch=FlowArchConfig, pairing=PairingConfig, train=FlowTrainConfig)

    def resolve(self, n_pairs_per_round: int) -> dict:
        spe = max(1, -(-n_pairs_per_round // self.train.flow_batch))
        return dict(steps_per_epoch=spe,
                    max_steps=int(round(self.train.epochs * spe)),
                    ckpt_every=max(1, int(round(self.train.ckpt_every_epochs * spe))))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FlowConfig":
        return cls(**{k: _build(kind, d.get(k)) for k, kind in cls.KINDS.items()})

    @classmethod
    def from_cli(cls, a) -> "FlowConfig":
        return cls(**_from_cli(cls.KINDS, a))

    def describe(self) -> str:
        return "\n".join(f"[flow.{g}] " + "  ".join(f"{k}={v}" for k, v in asdict(getattr(self, g)).items())
                         for g in self.KINDS)


# Per-dataset constants of the published recipe, applied to every run by `apply_dataset_defaults`.
# An explicit command-line flag overrides them.
DATASET_DEFAULTS = {
    "additive": {"vae": {"pert_noise": 0.4}, "flow": {"pert_noise": 0.4}},
    "holdout":  {"vae": {"pert_noise": 0.4}, "flow": {"pert_noise": 0.4}},
    "replogle": {"vae": {"pert_noise": 0.2}, "flow": {"pert_noise": 0.2, "cond_out_dropout": 0.9}},
    # Drugs are encoded by Morgan fingerprints.
    "combosciplex": {"vae": {"features": "morgan", "d_feat": 16, "ppi_feat": 0, "pathway_feat": 0,
                             "pert_noise": 0.4},
                     "flow": {"pert_noise": 0.4}},
}


def apply_dataset_defaults(namespace, stage: str, dataset: str) -> None:
    """Write DATASET_DEFAULTS[dataset][stage] onto an argparse namespace; explicit flags win."""
    for name, value in DATASET_DEFAULTS.get(dataset, {}).get(stage, {}).items():
        if getattr(namespace, name) is None:
            setattr(namespace, name, value)
