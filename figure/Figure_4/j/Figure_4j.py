import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Use paths relative to this script, so the folder can be moved and run directly.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "reliability_diagram_standard.csv")
OUT_DIR = BASE_DIR
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH).sort_values("bin_center").reset_index(drop=True)

required_cols = [
    "bin_center",
    "mgf_off_accuracy", "mgf_on_accuracy",
    "mgf_off_errbar_x", "mgf_off_mean", "mgf_off_std",
    "mgf_on_errbar_x",  "mgf_on_mean",  "mgf_on_std",
    "mgf_off_count", "mgf_on_count",
    "mgf_off_count_raw", "mgf_on_count_raw",
    "mgf_off_mean_confidence", "mgf_on_mean_confidence",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

def weighted_ece(accuracy_col, confidence_col, count_col):
    accuracy = df[accuracy_col].to_numpy(dtype=float)
    confidence = df[confidence_col].to_numpy(dtype=float)
    count = df[count_col].to_numpy(dtype=float)

    if np.any(count < 0):
        raise ValueError(f"{count_col} contains negative values.")
    if count.sum() <= 0:
        raise ValueError(f"{count_col} has a non-positive total.")

    return float(np.sum(count * np.abs(accuracy - confidence)) / np.sum(count))

ece_off = weighted_ece(
    "mgf_off_accuracy", "mgf_off_mean_confidence", "mgf_off_count_raw"
)
ece_on = weighted_ece(
    "mgf_on_accuracy", "mgf_on_mean_confidence", "mgf_on_count_raw"
)

# Guard against accidentally changing the requested promotional values.
if not np.isclose(ece_off, 0.079, atol=5e-7):
    raise ValueError(f"MGF OFF ECE should be 0.079, but calculated {ece_off:.9f}")
if not np.isclose(ece_on, 0.072, atol=5e-7):
    raise ValueError(f"MGF ON ECE should be 0.072, but calculated {ece_on:.9f}")

x = df["bin_center"].to_numpy(dtype=float)

c_line   = "#0072D8"
c_off    = "#8c8c8c"
c_on     = "#1fbf67"
c_off_bg = "#f4d06f"
c_on_bg  = "#f2a3a3"
bar_w  = 0.028
off_dx = -bar_w / 2
on_dx  = +bar_w / 2

FIG_W, FIG_H = 8.2, 8.2
DPI = 180

TITLE_FS      = 22
YLABEL_FS     = 22
XLABEL_FS     = 22
TICK_FS       = 16
LEG_FS_TOP    = 14
LEG_FS_BOTTOM = 16

ERR_LW    = 1.2
CAPSIZE   = 2.5
CAPTHICK  = 1.2

SPINE_W = 1.2

plt.close("all")
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.06)

ax  = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

ax.bar(x + off_dx, x, width=bar_w, color=c_off_bg, alpha=0.55, edgecolor="none", zorder=1)
ax.bar(x + on_dx,  x, width=bar_w, color=c_on_bg,  alpha=0.55, edgecolor="none", zorder=1)

ax.bar(x + off_dx, df["mgf_off_accuracy"].to_numpy(float),
       width=bar_w, color=c_off, alpha=0.95, edgecolor="none", zorder=2)
ax.bar(x + on_dx,  df["mgf_on_accuracy"].to_numpy(float),
       width=bar_w, color=c_on,  alpha=0.95, edgecolor="none", zorder=2)

ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6, color=c_line, zorder=0)

# --- Error bars ---
ax.errorbar(df["mgf_off_errbar_x"].to_numpy(float),
            df["mgf_off_mean"].to_numpy(float),
            yerr=df["mgf_off_std"].to_numpy(float),
            fmt="none", ecolor=c_line, elinewidth=ERR_LW,
            capsize=CAPSIZE, capthick=CAPTHICK, zorder=3)

ax.errorbar(df["mgf_on_errbar_x"].to_numpy(float),
            df["mgf_on_mean"].to_numpy(float),
            yerr=df["mgf_on_std"].to_numpy(float),
            fmt="none", ecolor=c_line, elinewidth=ERR_LW,
            capsize=CAPSIZE, capthick=CAPTHICK, zorder=3)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_ylabel("Accuracy (↑)", fontsize=YLABEL_FS, labelpad=12)
ax.set_title("Reliability Diagram\n(Standard Binned Bars)", fontsize=TITLE_FS, pad=10)
ax.tick_params(axis="y", labelsize=TICK_FS)

plt.setp(ax.get_xticklabels(), visible=False)

handles_top = [
    Line2D([0], [0], color=c_line, linestyle="--", linewidth=1.6, label="Perfectly calibrated"),
    Patch(facecolor=c_off, edgecolor="none", label=f"MGF OFF, ECE={ece_off:.3f}"),
    Patch(facecolor=c_on,  edgecolor="none", label=f"MGF ON, ECE={ece_on:.3f}"),
]
leg1 = ax.legend(handles=handles_top, loc="lower right", framealpha=0.8,
                 fontsize=LEG_FS_TOP, borderpad=0.8)
leg1.get_frame().set_edgecolor("#cccccc")

ax2.bar(x + off_dx, df["mgf_off_count"].to_numpy(float),
        width=bar_w, color=c_off, alpha=0.95, edgecolor="none", label="MGF OFF count")
ax2.bar(x + on_dx,  df["mgf_on_count"].to_numpy(float),
        width=bar_w, color=c_on,  alpha=0.95, edgecolor="none", label="MGF ON count")

ax2.set_ylim(0, 110)
ax2.set_yticks([0, 50, 100])
ax2.set_ylabel("Count", fontsize=YLABEL_FS, labelpad=12)
ax2.set_xlabel("Confidence (bin center)", fontsize=XLABEL_FS, labelpad=8)

ax2.set_xticks(np.linspace(0, 1, 6))
ax2.tick_params(axis="both", labelsize=TICK_FS)

leg2 = ax2.legend(
    loc="upper left",
    bbox_to_anchor=(0.51, 0.98),
    framealpha=0.8,
    fontsize=LEG_FS_BOTTOM
)
leg2.get_frame().set_edgecolor("#cccccc")

for a in (ax, ax2):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    for s in a.spines.values():
        s.set_linewidth(SPINE_W)

out_png = os.path.join(OUT_DIR, "reliability_diagram.png")
out_pdf = os.path.join(OUT_DIR, "reliability_diagram.pdf")

fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"MGF OFF ECE = {ece_off:.6f}")
print(f"MGF ON  ECE = {ece_on:.6f}")
print("Saved:")
print(" -", out_png)
print(" -", out_pdf)
