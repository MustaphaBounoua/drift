"""Train stage 1, the disentangled VAE.

    python scripts/train_vae.py --run-name additive-s0-vae --dataset additive --seed 0

Every flag is generated from the dataclasses in `drift/config.py`; their defaults, plus the
per-dataset constants in `DATASET_DEFAULTS`, are the published recipe.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import lightning as L
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from drift.config import (ArchConfig, DataConfig, ObjectiveConfig,  # noqa: E402
                          TrainConfig, VAEConfig, apply_dataset_defaults)
from drift.data import PerturbData  # noqa: E402
from drift.features import feature_table  # noqa: E402
from drift.vae import DriftVAE  # noqa: E402

GROUPS = dict(data=DataConfig, arch=ArchConfig, objective=ObjectiveConfig, train=TrainConfig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out", default="checkpoints")
    for gname, kind in GROUPS.items():
        g = p.add_argument_group(gname)
        for f in dataclasses.fields(kind):
            flag = "--" + f.name.replace("_", "-")
            if isinstance(f.default, bool):                 # --flag 0 / --flag 1
                g.add_argument(flag, type=int, choices=[0, 1], default=None)
            else:
                g.add_argument(flag, type=type(f.default), default=None)
    return p


def main():
    a = build_parser().parse_args()
    apply_dataset_defaults(a, "vae", a.dataset or DataConfig.dataset)
    cfg = VAEConfig.from_cli(a)
    L.seed_everything(cfg.train.seed, workers=True)

    data = PerturbData(cfg.data.data_dir, dataset=cfg.data.dataset, device="cuda")
    W = feature_table(data, cfg.data, device="cuda", verbose=True)

    steps = cfg.resolve(len(data.train.counts))
    out = pathlib.Path(a.out) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"cfg": cfg.to_dict(), "steps": steps}, indent=2))
    print(cfg.describe(), flush=True)
    print(f"[budget] {cfg.train.epochs} epochs x {steps['steps_per_epoch']} steps/epoch "
          f"= {steps['max_steps']} steps", flush=True)

    model = DriftVAE(data=data, gene_features=W, cfg=cfg)

    # one epoch = one pass over a random permutation of the training cells
    n, bs = len(data.train.counts), cfg.train.batch_size
    gen = torch.Generator(device="cuda").manual_seed(cfg.train.seed)
    tr = data.train

    def make(idx):
        return dict(counts=tr.counts[idx], expr=tr.expr[idx], g1=tr.g1[idx], g2=tr.g2[idx],
                    covar=tr.covar[idx])

    class Batches(torch.utils.data.IterableDataset):
        def __iter__(self):
            while True:
                perm = torch.randperm(n, device="cuda", generator=gen)
                for i in range(0, n - bs + 1, bs):
                    yield make(perm[i:i + bs])

    trainer = L.Trainer(
        max_steps=steps["max_steps"], accelerator="gpu", devices=1, logger=False,
        enable_checkpointing=True, enable_progress_bar=False,
        gradient_clip_val=(cfg.train.grad_clip or None),
        callbacks=[L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(out), every_n_train_steps=steps["ckpt_every"],
            save_top_k=0, save_last=True)],
        num_sanity_val_steps=0)
    trainer.fit(model, torch.utils.data.DataLoader(Batches(), batch_size=None))
    print("[done]", a.run_name, flush=True)


if __name__ == "__main__":
    main()
