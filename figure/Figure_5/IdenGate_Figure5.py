import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0) PATHS
# ============================================================
BASE_DIR = r"file_name"
CSV_NAME = "readerstudy_400rows.csv"

OUT_PNG = os.path.join(BASE_DIR, "Figure5_fourpanel.png")
OUT_PDF = os.path.join(BASE_DIR, "Figure5_fourpanel.pdf")

SAVE_PANELS = True
OUT_A = os.path.join(BASE_DIR, "Figure5_panel_a.png")
OUT_B = os.path.join(BASE_DIR, "Figure5_panel_b.png")
OUT_C = os.path.join(BASE_DIR, "Figure5_panel_c.png")
OUT_D = os.path.join(BASE_DIR, "Figure5_panel_d.png")

# ============================================================
# 1) BOOTSTRAP
# ============================================================
N_BOOT = 5000
ALPHA = 0.05

SEEDS = {
    "A1": 66,
    "A2": 271,
    "A0": 37,
    "B1": 113,
    "B2": 125,
    "B0": 1295,
}

def bootstrap_ci_mean(x, n_boot=N_BOOT, seed=0, alpha=ALPHA):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)

def make_pairs(df):
    key = ["repeat_id", "split_seed", "image_id", "set_id", "clinician_id", "gt_grade"]
    ai = df[df["condition"].str.contains("AI", case=False)].copy()
    no = df[df["condition"].str.contains("No", case=False)].copy()
    pair = ai.merge(no, on=key, suffixes=("_ai", "_no"))
    pair["d_conf"] = pair["confidence_score_ai"] - pair["confidence_score_no"]
    pair["d_time"] = pair["decision_time_sec_ai"] - pair["decision_time_sec_no"]
    pair["ai_prob"] = pair["ai_shown_prob_top1_ai"]
    return pair

def get_conf_for_calibration(df_sub):
    if "confidence_prob_calib" in df_sub.columns:
        return np.clip(df_sub["confidence_prob_calib"].to_numpy(dtype=float), 0, 1)
    return np.clip(df_sub["confidence_score"].to_numpy(dtype=float) / 100.0, 0, 1)

def bin_stats_meanconf(conf01, correct, n_bins=10):
    conf01 = np.asarray(conf01, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)

    mean_conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf01 >= lo) & (conf01 < hi) if i < n_bins - 1 else (conf01 >= lo) & (conf01 <= hi)
        cnt[i] = int(m.sum())
        if cnt[i] > 0:
            mean_conf[i] = float(conf01[m].mean())
            acc[i] = float(correct[m].mean())
    return mean_conf, acc, cnt

def ece_from_bin_meanconf(mean_conf, acc, cnt):
    cnt = np.asarray(cnt, dtype=int)
    N = cnt.sum()
    ece = 0.0
    for mc, a, c in zip(mean_conf, acc, cnt):
        if c > 0:
            ece += (c / N) * abs(a - mc)
    return float(ece)

def safety_audit(pair, thr):
    sub = pair[pair["ai_prob"] >= thr].copy()
    n = len(sub)
    c2w = int(((sub["correct_no"] == 1) & (sub["correct_ai"] == 0)).sum())
    w2c = int(((sub["correct_no"] == 0) & (sub["correct_ai"] == 1)).sum())
    net_pp = (c2w - w2c) / n * 100.0 if n else 0.0
    dconf = float(sub["d_conf"].mean()) if n else np.nan

    n_ai = len(pair)
    prev = (n / n_ai) * 100.0 if n_ai else 0.0
    return dict(n=n, c2w=c2w, w2c=w2c, net=net_pp, dconf=dconf, prev=prev)

def rater_stats(pair, rater_id, col, seed):
    x = pair.loc[pair["clinician_id"] == rater_id, col].to_numpy()
    m = float(x.mean())
    lo, hi = bootstrap_ci_mean(x, n_boot=N_BOOT, seed=seed, alpha=ALPHA)
    return m, lo, hi

def overall_stats(pair, col, seed):
    x = pair.groupby("image_id")[col].mean().to_numpy()
    m = float(x.mean())
    lo, hi = bootstrap_ci_mean(x, n_boot=N_BOOT, seed=seed, alpha=ALPHA)
    return m, lo, hi

# ============================================================
# 3) load data
# ============================================================
csv_path = os.path.join(BASE_DIR, CSV_NAME)
df = pd.read_csv(csv_path)

need_cols = [
    "condition", "confidence_score", "decision_time_sec", "ai_shown_prob_top1",
    "correct", "repeat_id", "split_seed", "image_id", "set_id", "clinician_id", "gt_grade"
]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}")

pair = make_pairs(df)

# ============================================================
# 4) compute stats
# ============================================================
A1 = rater_stats(pair, 1, "d_conf", seed=SEEDS["A1"])
A2 = rater_stats(pair, 2, "d_conf", seed=SEEDS["A2"])
A0 = overall_stats(pair, "d_conf", seed=SEEDS["A0"])

B1 = rater_stats(pair, 1, "d_time", seed=SEEDS["B1"])
B2 = rater_stats(pair, 2, "d_time", seed=SEEDS["B2"])
B0 = overall_stats(pair, "d_time", seed=SEEDS["B0"])

una = df[df["condition"] == "No-assist"].copy()
aia = df[df["condition"] == "AI-assist"].copy()

conf_u = get_conf_for_calibration(una)
conf_a = get_conf_for_calibration(aia)

mc_u, acc_u, cnt_u = bin_stats_meanconf(conf_u, una["correct"].to_numpy(), n_bins=10)
mc_a, acc_a, cnt_a = bin_stats_meanconf(conf_a, aia["correct"].to_numpy(), n_bins=10)
ece_u = ece_from_bin_meanconf(mc_u, acc_u, cnt_u)
ece_a = ece_from_bin_meanconf(mc_a, acc_a, cnt_a)

mask_u = (cnt_u > 0)
mask_a = (cnt_a > 0)

audit70 = safety_audit(pair, 0.70)
audit80 = safety_audit(pair, 0.80)

# ============================================================
# 5) STYLE + AXIS SCALES
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.0,
})

CI_LW = 2.2
DOT_MS = 5.5
VLINE_LW = 1.0
VLINE_ALPHA = 0.75

COL_R1 = "#4FA7D1"
COL_R2 = "#C00000"
COL_OV = "#2CA02C"
COL_GRAY = "#7F7F7F"

CAL_LW = 1.8
CAL_MS = 5.5
PERF_LW = 1.0

fig = plt.figure(figsize=(12.0, 7.0))
gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.62)

ypos = [2, 1, 0]
ylabels = ["Rater 1", "Rater 2", "Overall"]
colors = [COL_R1, COL_R2, COL_OV]

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.0, length=4)

# ============================================================
# 6) draw panels
# ============================================================

# ---------- Panel a ----------
ax1 = fig.add_subplot(gs[0, 0])
ax1.axvline(0, ls="--", lw=VLINE_LW, c="gray", alpha=VLINE_ALPHA)
for y, col, (m, lo, hi) in zip(ypos, colors, [A1, A2, A0]):
    ax1.plot([lo, hi], [y, y], lw=CI_LW, c=col, solid_capstyle="butt")
    ax1.plot([m], [y], marker="o", ms=DOT_MS, c=col)
    ax1.text(hi + 0.10, y, f"{m:+.2f} ({lo:+.2f}, {hi:+.2f})",
             va="center", ha="left", clip_on=False)

ax1.set_yticks(ypos, ylabels)
ax1.set_title("Confidence gain", pad=6)
ax1.set_xlabel("Δ Confidence (AI-assisted − Unaided), points", labelpad=4)

ax1.set_xlim(-2, 13)
ax1.set_xticks([-2, 0, 2, 4, 6, 8])

ax1.text(-0.19, 1.02, "a", transform=ax1.transAxes, fontweight="bold", fontsize=13)
clean_axes(ax1)

# ---------- Panel b ----------
ax2 = fig.add_subplot(gs[0, 1])
ax2.axvline(0, ls="--", lw=VLINE_LW, c="gray", alpha=VLINE_ALPHA)

for y, col, (m, lo, hi) in zip(ypos, colors, [B1, B2, B0]):
    ax2.plot([lo, hi], [y, y], lw=CI_LW, c=col, solid_capstyle="butt")
    ax2.plot([m], [y], marker="o", ms=DOT_MS, c=col)
    ax2.text(hi + 0.06, y, f"{m:+.2f}s ({lo:+.2f}, {hi:+.2f})", va="center")

ax2.set_yticks(ypos, ylabels)
ax2.set_title("Decision time change", pad=6)
ax2.set_xlabel("Δ Time (AI-assisted − Unaided), seconds", labelpad=4)

ax2.set_xlim(-1.6, 3.3)
ax2.set_xticks([-1, 0, 1])

ax2.text(-0.19, 1.02, "b", transform=ax2.transAxes, fontweight="bold", fontsize=13)
clean_axes(ax2)

# ---------- Panel c ----------
ax3 = fig.add_subplot(gs[1, 0])

line_perf, = ax3.plot([0, 1], [0, 1], ls="--", lw=PERF_LW, c=COL_OV, label="Perfect calibration")
line_u, = ax3.plot(mc_u[mask_u], acc_u[mask_u], marker="o", ms=CAL_MS, lw=CAL_LW, c=COL_GRAY,
                   label=f"Unaided (ECE={ece_u*100:.1f}%)")
line_a, = ax3.plot(mc_a[mask_a], acc_a[mask_a], marker="o", ms=CAL_MS, lw=CAL_LW, c=COL_OV,
                   label=f"AI-assisted (ECE={ece_a*100:.1f}%)")

ax3.set_xlim(0, 1.09)
ax3.set_ylim(0, 1)

ticks01 = np.linspace(0, 1, 6)
ax3.set_xticks(ticks01)
ax3.set_yticks(ticks01)

ax3.set_xlabel("Mean confidence", labelpad=4)
ax3.set_ylabel("Empirical accuracy", labelpad=4)
ax3.set_title("Clinician confidence–accuracy calibration\n(10-bin ECE)", pad=6, loc="left")

handles = [line_u, line_a, line_perf]
labels = [h.get_label() for h in handles]
ax3.legend(handles, labels, loc="lower right", frameon=False, handlelength=2.2)

ax3.text(-0.19, 1.02, "c", transform=ax3.transAxes, fontweight="bold", fontsize=13)
clean_axes(ax3)
ax4 = fig.add_subplot(gs[1, 1])
prev = [audit70["prev"], audit80["prev"]]
ax4.bar([0, 1], prev, color=COL_OV, alpha=0.80, width=0.72)

ax4.set_xticks([0, 1], ["AI top-1\nprob. ≥ 0.70", "AI top-1\nprob. ≥ 0.80"])
ax4.set_ylabel("Prevalence among AI-\nassisted reads (%)", labelpad=4)
ax4.set_title("Safety audit (automation risk)", pad=10)
ax4.text(-0.19, 1.02, "d", transform=ax4.transAxes, fontweight="bold", fontsize=13)

ax4.set_ylim(0, 7.35)
ax4.set_yticks(range(0, 8, 1))
clean_axes(ax4)

def annotate_bar(x, info, bar_h, y_offset=0.10, y_cap=6.25):
    txt = (
        f"n = {info['n']}\n"
        f"C→W = {info['c2w']}/{info['n']} ({(info['c2w']/info['n']*100 if info['n'] else 0):.1f}%)\n"
        f"W→C = {info['w2c']}/{info['n']} ({(info['w2c']/info['n']*100 if info['n'] else 0):.1f}%)\n"
        f"Net harm = {info['net']:+.1f} pp\n"
        f"Δ Conf = {info['dconf']:+.1f} pts"
    )
    y = min(bar_h + y_offset, y_cap)
    ax4.text(x, y, txt, ha="center", va="bottom", fontsize=10.5)

annotate_bar(0, audit70, prev[0])
annotate_bar(1, audit80, prev[1])


# ============================================================
# 7) SAVE
# ============================================================
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)

if SAVE_PANELS:
    def save_panel(ax, out_path):
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, dpi=300, bbox_inches=bb)
        print("Saved panel:", out_path)

    fig.canvas.draw()
    save_panel(ax1, OUT_A)
    save_panel(ax2, OUT_B)
    save_panel(ax3, OUT_C)
    save_panel(ax4, OUT_D)

print("\n=== Numbers computed from CSV ===")
print("Panel a (ΔConfidence):")
print("  Rater1:", A1)
print("  Rater2:", A2)
print("  Overall:", A0)
print("Panel b (ΔTime):")
print("  Rater1:", B1)
print("  Rater2:", B2)
print("  Overall:", B0)
print(f"Panel c ECE: unaided={ece_u*100:.3f}%, AI-assisted={ece_a*100:.3f}%")
print("Panel d audit >=0.70:", audit70)
print("Panel d audit >=0.80:", audit80)
