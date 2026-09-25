"""Score a trained model on the test split and write the metrics to JSON.

    python scripts/evaluate.py checkpoints/additive-s0/last.ckpt --json results/additive-s0.json

Norman and Replogle are scored with the scBIG metrics, ComboSciPlex with cell-eval.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from drift.config import VAEConfig  
from drift.data import PerturbData  
from drift.eval import score_celleval, score_genetic  
from drift.flow import DriftFlow 
from drift.vae import DriftVAE  


def load(ckpt: str):
    meta = json.load(open(pathlib.Path(ckpt).parent / "config.json"))
    vcfg = VAEConfig.from_dict(meta["vae"])
    data = PerturbData(vcfg.data.data_dir, dataset=vcfg.data.dataset, device="cuda")
    vae = DriftVAE.from_checkpoint(meta["vae_ckpt"], data).eval()
    flow = DriftFlow.load_from_checkpoint(ckpt, data=data, vae=vae, map_location="cuda").eval()
    flow.setup()
    return flow


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="stage-2 checkpoint; its config.json names the stage-1 checkpoint")
    ap.add_argument("--json", default=None, help="write the metrics here")
    a = ap.parse_args()

    flow = load(a.ckpt)
    dataset = flow.data.dataset
    if dataset == "combosciplex":
        res = {"test": score_celleval(flow)}
        rows = [("test", res["test"])]
    else:
        res = score_genetic(flow)
        rows = sorted(res.items())

    print(f"\n{pathlib.Path(a.ckpt).parent.name}  ({dataset}, test split)")
    for name, m in rows:
        print(f"  {name:<16} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(
            {"ckpt": a.ckpt, "dataset": dataset, "metrics": res}, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
