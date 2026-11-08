"""Runs ALL ablation comparisons in order and collects every result in <out>/results/. Calibration split only (no test look is used).
Prerequisite: `p10_ablations.py --phase calib` has finished (23 variants frozen under <out>/variants).
  python scripts/p10_run_all.py --p5q artifacts/p5q --out artifacts/ablations
Writes in <out>/results/: ablation_calibration.csv, gap_report_calibration.csv, decision_table_calibration.csv, decision_table_full.md, comparison_all.md, SUMMARY.txt"""
import sys, subprocess, argparse, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--shm", default="/dev/shm/aafv/abl")
ap.add_argument("--betas", default="0.005,0.01,0.02"); ap.add_argument("--subset", default=""); ap.add_argument("--tag", default="")
a = ap.parse_args(); out = Path(a.out); res = out / ("results" + a.tag); res.mkdir(parents=True, exist_ok=True)
import os
if a.subset: os.environ["AAFV_SUBSET"] = a.subset          # inherited by the sub-scripts: every variant is then evaluated on the same pairs
import p10_ablations as A
names = [n for n in A.V if (out / "variants" / n / "frozen.pkl").exists()]
missing = [n for n in A.V if n not in names]
print(f"{len(names)} frozen variants found" + (f"; MISSING: {missing}" if missing else "")); v = ",".join(names)
def run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], capture_output=True, text=True)
    print(script, "ok" if r.returncode == 0 else "FAILED\n" + r.stderr[-1500:]); return r.returncode == 0
ok = run("p10_gap_report.py", "--p5q", a.p5q, "--out", a.out, "--shm", a.shm, "--variants", v)
ok &= run("p10_decision_table.py", "--p5q", a.p5q, "--out", a.out, "--shm", a.shm, "--variants", v, "--betas", a.betas)
ok &= run("p10_compare_report.py", "--out", a.out)   # reads gap_report_calibration.csv / decision_table_calibration.csv just written
d = pd.read_csv(out / "decision_table_calibration.csv"); md = ["# Decision table: share of pairs in each zone (calibration split, development numbers)",
    "Accept threshold = frozen zero-false-accept rule. Reject threshold = the beta-quantile of same-person scores (beta = target false-reject rate).",
    "Different-person pairs auto-accepted (false accepts) are counts (0 by construction on calibration). The comparison between models is in the D_*: reject columns.", ""]
for beta, g in d.groupby("beta"):
    t = pd.DataFrame({"variant": g.variant, "T_acc": g.T_accept, "T_rej": g.T_reject, "same: accept": g.same_acc, "same: review": g.same_rev, "same: reject": g.same_rej,
        "D_random: reject": g.D_random_rej, "D_random: review": g.D_random_rev, "D_random: FA": g.D_random_FA,
        "D_all_random: reject": g.D_all_random_rej, "D_all_random: review": g.D_all_random_rev, "D_all_random: FA": g.D_all_random_FA,
        "D_hard: reject": g.D_hard_rej, "D_hard: review": g.D_hard_rev, "D_hard: FA": g.D_hard_FA, "automation": g.automation_balanced}).round(3)
    md += [f"## false-reject target beta = {beta}", "```", t.to_string(index=False), "```", ""]
(res / "decision_table_full.md").write_text("\n".join(md))
for f in ("ablation_calibration.csv", "gap_report_calibration.csv", "decision_table_calibration.csv", "comparison_all.md"):
    if (out / f).exists(): shutil.copy(out / f, res / f)
head = ["SUMMARY (calibration split, development numbers)", f"variants: {len(names)}", ""]
c = (res / "comparison_all.md").read_text().splitlines() if (res / "comparison_all.md").exists() else []
head += [x for x in c if x.startswith("## ") or x.startswith("better:")]
(res / "SUMMARY.txt").write_text("\n".join(head)); print("\n".join(head)); print("\nAll files in", res); print(sorted(p.name for p in res.iterdir()))
sys.exit(0 if ok else 1)