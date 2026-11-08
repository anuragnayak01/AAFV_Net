"""Builds comparison_all.md from the gap report and decision table CSVs: for each reference model, every metric is listed as better / no clear difference / worse.
Gap-bin metrics use the bootstrap intervals (non-overlapping = clear); decision-table metrics have no interval, so a difference above 0.02 is called clear.
  python scripts/p10_compare_report.py --out artifacts/ablations [--ours full] [--refs arcface_only,direct5_logreg]"""
import argparse, re
from pathlib import Path
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ours", default="full"); ap.add_argument("--refs", default="arcface_only,direct5_logreg,direct_only_mlp,baseline_siamese_cnn,baseline_siamese_vit,baseline_vlm"); a = ap.parse_args(); out = Path(a.out); gap = pd.read_csv(out / "gap_report_calibration.csv"); dec = pd.read_csv(out / "decision_table_calibration.csv")
def parse(x):
    m = re.match(r"([\d.]+) \[([\d.]+),([\d.]+)\]", str(x)); return tuple(map(float, m.groups())) if m else None
lines = ["# Complete comparison (calibration split, development numbers)", "", f"Ours = `{a.ours}`. Every metric is listed; nothing is filtered. 'better' = clearly higher for ours; 'worse' = clearly lower.", ""]
for ref in [r for r in a.refs.split(",") if r in set(gap.variant)]:
    res = {"better": [], "no clear difference": [], "worse": []}
    for metric in [c for c in gap.columns if c == "AUC" or c.startswith("TPR")]:
        for b in ["G1", "G2", "G3", "G4", "G5", "G6"]:
            o = parse(gap[(gap.variant == a.ours) & (gap.bin == b)][metric].squeeze()); r = parse(gap[(gap.variant == ref) & (gap.bin == b)][metric].squeeze())
            if o is None or r is None: continue
            v = "better" if o[1] > r[2] else "worse" if o[2] < r[1] else "no clear difference"
            res[v].append(f"{metric}, gap bin {b}: ours {o[0]:.3f} vs {r[0]:.3f}")
    for beta in sorted(dec.beta.unique()):
        d = dec[dec.beta == beta].set_index("variant")
        if a.ours in d.index and ref in d.index:
            for m in ["same_acc", "same_acc_G4G6", "D_random_rej", "D_all_random_rej", "D_hard_rej", "automation_balanced"]:
                o, r = float(d.loc[a.ours, m]), float(d.loc[ref, m]); v = "better" if o - r > 0.02 else "worse" if r - o > 0.02 else "no clear difference"
                res[v].append(f"decision table (false-reject target {beta}), {m}: ours {o:.3f} vs {r:.3f}")
    lines += [f"## `{a.ours}` vs `{ref}`", f"better: {len(res['better'])} | no clear difference: {len(res['no clear difference'])} | worse: {len(res['worse'])}", ""]
    for k in ("better", "no clear difference", "worse"): lines += [f"### {k}"] + [f"- {x}" for x in res[k]] + [""]
(out / "comparison_all.md").write_text("\n".join(lines)); print("\n".join(lines))