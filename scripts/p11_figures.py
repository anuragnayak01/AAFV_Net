"""Figures from results_subset: python scripts/p11_figures.py artifacts/ablations/results_subset [beta=0.01]"""
import sys; from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(sys.argv[1]); beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01; F = R / "figs"; F.mkdir(exist_ok=True)
dt = pd.read_csv(R / "decision_table_calibration.csv"); gp = pd.read_csv(R / "gap_report_calibration.csv")
M = ["full", "arcface_only", "direct_only_mlp", "baseline_siamese_cnn", "baseline_siamese_vit", "baseline_vlm"]; M = [m for m in M if m in set(dt.variant)]
LB = {"full": "AAFV-Net (ours)", "arcface_only": "ArcFace only", "direct_only_mlp": "ours, direct only", "baseline_siamese_cnn": "Siamese CNN", "baseline_siamese_vit": "Siamese ViT", "baseline_vlm": "VLM (Unsloth)"}
col = dict(zip(M, ["#1f6feb", "#8b949e", "#7ea6e0", "#d29922", "#cf5c36", "#8250df"]))
# Fig 1: what happens to same-person pairs and to hard impostors (zero false accepts, false-reject = beta)
d = dt[(dt.beta == beta) & dt.variant.isin(M)].set_index("variant").loc[M]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(M))
for a_, (title, cols, names) in zip(ax, [("Same-person pairs", ["same_acc", "same_rev", "same_rej"], ["auto-accept", "review", "auto-reject (false reject)"]), ("Hard different-person pairs (D_hard)", ["D_hard_FA", "D_hard_rev", "D_hard_rej"], ["false accept", "review", "auto-reject"])]):
    b = np.zeros(len(M))
    for c, n, k in zip(cols, names, ["#2da44e", "#d4a72c", "#cf222e"] if cols[0] == "same_acc" else ["#cf222e", "#d4a72c", "#2da44e"]): a_.bar(x, d[c], bottom=b, label=n, color=k); b += d[c].to_numpy()
    a_.set_xticks(x); a_.set_xticklabels([LB[m] for m in M], rotation=30, ha="right"); a_.set_title(title); a_.set_ylim(0, 1); a_.legend(fontsize=8, loc="lower right")
fig.suptitle(f"Zero false accepts on calibration, false-reject target {beta}"); fig.tight_layout(); fig.savefig(F / "fig1_decision_outcomes.png", dpi=200); plt.close(fig)
# Fig 2: hard impostors rejected and balanced automation versus the false-reject target
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for m in M:
    s = dt[dt.variant == m].sort_values("beta")
    ax[0].plot(s.beta, s.D_hard_rej, "o-", color=col[m], label=LB[m]); ax[1].plot(s.beta, s.automation_balanced, "o-", color=col[m], label=LB[m])
for a_, t in zip(ax, ["D_hard auto-rejected", "Balanced automation rate"]): a_.set_xlabel("false-reject target"); a_.set_title(t); a_.set_ylim(0, 1.02); a_.set_xscale("log")
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(F / "fig2_vs_false_reject_target.png", dpi=200); plt.close(fig)
# Fig 3: AUC by age-gap bin (bins with too few pairs are dropped)
g = gp[gp.variant.isin(M)].copy(); g["AUC"] = pd.to_numeric(g["AUC"], errors="coerce"); g = g.dropna(subset=["AUC"])
if len(g):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in M:
        s = g[g.variant == m].sort_values("bin")
        if len(s): ax.plot(s.bin, s.AUC, "o-", color=col[m], label=LB[m])
    ax.set_ylabel("AUC"); ax.set_xlabel("age-gap bin"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(F / "fig3_auc_by_gap.png", dpi=200); plt.close(fig)
print("saved to", F, sorted(p.name for p in F.iterdir()))