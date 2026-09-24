"""Train stage 2, the conditional flow, on a frozen stage-1 checkpoint.

    python scripts/train_flow.py --run-name additive-s0 --seed 0 \
        --vae-ckpt checkpoints/additive-s0-vae/last.ckpt

The dataset is read from the VAE checkpoint. Every flag is generated from the dataclasses in
`drift/config.py`; their defaults, plus `DATASET_DEFAULTS`, are the published recipe.
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

from drift.config import (FlowArchConfig, FlowConfig, FlowTrainConfig,  # noqa: E402
                          PairingConfig, VAEConfig, apply_dataset_defaults)
from drift.data import PerturbData  # noqa: E402
from drift.flow import DriftFlow  # noqa: E402
from drift.vae import DriftVAE  # noqa: E402

GROUPS = dict(arch=FlowArchConfig, pairing=PairingConfig, train=FlowTrainConfig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--out", default="checkpoints")
    for gname, kind in GROUPS.items():
        g = p.add_argument_group(gname)
        for f in dataclasses.fields(kind):
            flag = "--" + f.name.replace("_", "-")
            if isinstance(f.default, bool):
                g.add_argument(flag, type=int, choices=[0, 1], default=None)
            else:
                g.add_argument(flag, type=type(f.default), default=None)
    return p


def main():
    a = build_parser().parse_args()
    cfg = FlowConfig.from_cli(a)
    L.seed_everything(cfg.train.seed, workers=True)

    hp = torch.load(a.vae_ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    vcfg = VAEConfig.from_dict(hp["cfg"])
    apply_dataset_defaults(a, "flow", vcfg.data.dataset)
    cfg = FlowConfig.from_cli(a)

    data = PerturbData(vcfg.data.data_dir, dataset=vcfg.data.dataset,
                       device="cuda")
    vae = DriftVAE.from_checkpoint(a.vae_ckpt, data).eval()

    flow = DriftFlow(data=data, vae=vae, cfg=cfg)
    flow.setup()
    steps = cfg.resolve(cfg.pairing.n_cond * cfg.pairing.block)

    out = pathlib.Path(a.out) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"flow": cfg.to_dict(), "vae_ckpt": a.vae_ckpt, "vae": vcfg.to_dict(), "steps": steps},
        indent=2))
    print(cfg.describe(), flush=True)
    print(f"[budget] {cfg.train.epochs} OT rounds x {steps['steps_per_epoch']} steps/round "
          f"= {steps['max_steps']} steps", flush=True)

    trainer = L.Trainer(
        max_steps=steps["max_steps"], accelerator="gpu", devices=1, logger=False,
        enable_progress_bar=False, gradient_clip_val=(cfg.train.grad_clip or None),
        callbacks=[L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(out), every_n_train_steps=steps["ckpt_every"],
            save_top_k=0, save_last=True)],
        num_sanity_val_steps=0)
    trainer.fit(flow)
    print("[done]", a.run_name, flush=True)


if __name__ == "__main__":
    main()
