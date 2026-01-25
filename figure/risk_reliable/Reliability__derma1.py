import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


IDENGATE_PATHS = [
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/1/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/2/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/3/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/4/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/42/retina_alme_seed42_preds_for_rc.npz"),
]
BASELINE_PATHS = [
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/1/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/2/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/3/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/4/retina_alme_seed42_preds_for_rc.npz"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/42/retina_alme_seed42_preds_for_rc.npz"),
]

IDENGATE_MAIN = Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/42/retina_alme_seed42_preds_for_rc.npz")
BASELINE_MAIN = Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/4/retina_alme_seed42_preds_for_rc.npz")

OUT_DIR = Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


N_BINS = 15
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
EPS = 1e-12


def load_conf_correct(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    if "conf" not in d.files:
        raise KeyError(f"'conf' not found in {npz_path}. keys={d.files}")

    conf = d["conf"].astype(np.float64).reshape(-1)

    if "correct" in d.files:
        correct = d["correct"].astype(np.float64).reshape(-1)
    else:
        if ("y_true" not in d.files) or ("y_pred" not in d.files):
            raise KeyError(f"Need 'correct' or ('y_true','y_pred') in {npz_path}. keys={d.files}")
        y_true = d["y_true"].reshape(-1)
        y_pred = d["y_pred"].reshape(-1)
        correct = (y_true == y_pred).astype(np.float64)

    if conf.shape[0] != correct.shape[0]:
        raise ValueError(f"Shape mismatch in {npz_path}: conf={conf.shape}, correct={correct.shape}")

    if np.nanmin(conf) < -1e-6 or np.nanmax(conf) > 1.0 + 1e-6:
        raise ValueError(f"[BAD conf range] {npz_path}: min={conf.min()}, max={conf.max()}")
    if np.nanmin(correct) < -1e-6 or np.nanmax(correct) > 1.0 + 1e-6:
        raise ValueError(f"[BAD correct range] {npz_path}: min={correct.min()}, max={correct.max()}")

    return conf, correct


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bin_edges: np.ndarray):

    conf = conf.astype(np.float64).reshape(-1)
    correct = correct.astype(np.float64).reshape(-1)

    conf_clamped = np.minimum(conf, 1.0 - EPS)
    bin_idx = np.digitize(conf_clamped, bin_edges, right=False) - 1

    n_bins = len(bin_edges) - 1
    bin_conf_mean = np.full(n_bins, np.nan, dtype=np.float64)
    bin_acc = np.full(n_bins, np.nan, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        m = (bin_idx == b)
        c = int(np.sum(m))
        bin_count[b] = c
        if c > 0:
            bin_conf_mean[b] = float(np.mean(conf[m]))
            bin_acc[b] = float(np.mean(correct[m]))

    N = float(len(conf))
    w = bin_count.astype(np.float64) / max(N, 1.0)
    diff = np.abs(np.nan_to_num(bin_acc) - np.nan_to_num(bin_conf_mean))
    ece = float(np.sum(w * diff))
    return bin_conf_mean, bin_acc, bin_count, ece


def save_bins_csv(path: Path, bin_edges: np.ndarray, bin_conf_mean, bin_acc, bin_count, ece: float):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "bin_conf_mean", "bin_acc", "bin_count"])
        for i in range(len(bin_edges) - 1):
            w.writerow([
                f"{bin_edges[i]:.6f}",
                f"{bin_edges[i+1]:.6f}",
                "" if np.isnan(bin_conf_mean[i]) else f"{float(bin_conf_mean[i]):.10f}",
                "" if np.isnan(bin_acc[i]) else f"{float(bin_acc[i]):.10f}",
                str(int(bin_count[i])),
            ])
        w.writerow([])
        w.writerow(["ECE", f"{ece:.10f}"])


def tcrit_975(df: int) -> float:

    if df <= 0:
        return 0.0
    if df == 1:
        return 12.7062047364
    if df == 2:
        return 4.3026527299
    if df == 3:
        return 3.1824463053
    if df == 4:
        return 2.7764451052
    return 1.96


def mean_ci_across_seeds(arr_2d: np.ndarray):

    S, B = arr_2d.shape
    mean = np.nanmean(arr_2d, axis=0)

    lo = np.full(B, np.nan, dtype=np.float64)
    hi = np.full(B, np.nan, dtype=np.float64)

    for b in range(B):
        vals = arr_2d[:, b]
        vals = vals[~np.isnan(vals)]
        s = len(vals)
        if s <= 1:
            lo[b] = mean[b]
            hi[b] = mean[b]
            continue
        std = float(np.std(vals, ddof=1))
        half = tcrit_975(s - 1) * std / np.sqrt(s)
        lo[b] = mean[b] - half
        hi[b] = mean[b] + half

    return mean, lo, hi


def mean_ci_1d(values):
    v = np.asarray(values, dtype=np.float64)
    m = float(np.mean(v))
    if len(v) <= 1:
        return m, m, m
    std = float(np.std(v, ddof=1))
    half = tcrit_975(len(v) - 1) * std / np.sqrt(len(v))
    return m, m - half, m + half


def main():

    conf_bl_main, corr_bl_main = load_conf_correct(BASELINE_MAIN)
    bl_conf_main, bl_acc_main, bl_cnt_main, bl_ece_main = reliability_bins(conf_bl_main, corr_bl_main, BIN_EDGES)

    conf_id_main, corr_id_main = load_conf_correct(IDENGATE_MAIN)
    id_conf_main, id_acc_main, id_cnt_main, id_ece_main = reliability_bins(conf_id_main, corr_id_main, BIN_EDGES)


    id_conf_list, id_acc_list, id_ece_list, id_seed_tags = [], [], [], []
    for p in IDENGATE_PATHS:
        conf, corr = load_conf_correct(p)
        bconf, bacc, bcnt, ece = reliability_bins(conf, corr, BIN_EDGES)
        seed_tag = p.parent.name
        id_seed_tags.append(seed_tag)
        id_conf_list.append(bconf)
        id_acc_list.append(bacc)
        id_ece_list.append(ece)
        save_bins_csv(OUT_DIR / f"Reliability_bins_idengate_seed{seed_tag}.csv",
                      BIN_EDGES, bconf, bacc, bcnt, ece)

    bl_conf_list, bl_acc_list, bl_ece_list, bl_seed_tags = [], [], [], []
    for p in BASELINE_PATHS:
        conf, corr = load_conf_correct(p)
        bconf, bacc, bcnt, ece = reliability_bins(conf, corr, BIN_EDGES)
        seed_tag = p.parent.name
        bl_seed_tags.append(seed_tag)
        bl_conf_list.append(bconf)
        bl_acc_list.append(bacc)
        bl_ece_list.append(ece)
        save_bins_csv(OUT_DIR / f"Reliability_bins_baseline_seed{seed_tag}.csv",
                      BIN_EDGES, bconf, bacc, bcnt, ece)

    id_conf_mat = np.stack(id_conf_list, axis=0)
    id_acc_mat = np.stack(id_acc_list, axis=0)
    bl_conf_mat = np.stack(bl_conf_list, axis=0)
    bl_acc_mat = np.stack(bl_acc_list, axis=0)

    id_acc_mean, id_acc_lo, id_acc_hi = mean_ci_across_seeds(id_acc_mat)
    bl_acc_mean, bl_acc_lo, bl_acc_hi = mean_ci_across_seeds(bl_acc_mat)

    bl_ece_mean, bl_ece_lo, bl_ece_hi = mean_ci_1d(bl_ece_list)
    id_ece_mean, id_ece_lo, id_ece_hi = mean_ci_1d(id_ece_list)

    ece_csv = OUT_DIR / "Reliability_RetinaMNIST_ECE.csv"
    with open(ece_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "ECE"])
        for seed, ece in zip(bl_seed_tags, bl_ece_list):
            w.writerow(["Baseline", seed, f"{ece:.10f}"])
        for seed, ece in zip(id_seed_tags, id_ece_list):
            w.writerow(["IdenGate", seed, f"{ece:.10f}"])
        w.writerow([])
        w.writerow(["model", "ECE_mean", "ECE_ci_low", "ECE_ci_high"])
        w.writerow(["Baseline", f"{bl_ece_mean:.10f}", f"{bl_ece_lo:.10f}", f"{bl_ece_hi:.10f}"])
        w.writerow(["IdenGate", f"{id_ece_mean:.10f}", f"{id_ece_lo:.10f}", f"{id_ece_hi:.10f}"])

    bin_left = BIN_EDGES[:-1]
    bin_right = BIN_EDGES[1:]
    bin_width = (bin_right - bin_left)
    bin_centers = (bin_left + bin_right) / 2.0

    x = bin_centers
    conf_ref = x

    bl_acc_main = np.clip(bl_acc_main, 0.0, 1.0)
    id_acc_main = np.clip(id_acc_main, 0.0, 1.0)
    bl_acc_mean = np.clip(bl_acc_mean, 0.0, 1.0)
    id_acc_mean = np.clip(id_acc_mean, 0.0, 1.0)
    bl_acc_lo = np.clip(bl_acc_lo, 0.0, 1.0)
    bl_acc_hi = np.clip(bl_acc_hi, 0.0, 1.0)
    id_acc_lo = np.clip(id_acc_lo, 0.0, 1.0)
    id_acc_hi = np.clip(id_acc_hi, 0.0, 1.0)

    m_bl = ~np.isnan(bl_acc_main)
    m_id = ~np.isnan(id_acc_main)
    m_ci_bl = ~np.isnan(bl_acc_mean)
    m_ci_id = ~np.isnan(id_acc_mean)

    bl_yerr = np.vstack([(bl_acc_mean - bl_acc_lo), (bl_acc_hi - bl_acc_mean)])
    id_yerr = np.vstack([(id_acc_mean - id_acc_lo), (id_acc_hi - id_acc_mean)])
    bl_yerr = np.clip(bl_yerr, 0.0, 1.0)
    id_yerr = np.clip(id_yerr, 0.0, 1.0)

    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfectly calibrated")

    offset = 0.18 * bin_width
    w = 0.38 * bin_width

    ax.bar(x[m_bl] - offset[m_bl], bl_acc_main[m_bl],
           width=w[m_bl], align="center", alpha=0.85,
           label="Baseline")

    bl_gap = np.clip(conf_ref - bl_acc_main, 0.0, 1.0)
    ax.bar(x[m_bl] - offset[m_bl], bl_gap[m_bl],
           bottom=bl_acc_main[m_bl], width=w[m_bl], align="center", alpha=0.25)

    ax.errorbar(x[m_ci_bl] - offset[m_ci_bl], bl_acc_mean[m_ci_bl],
                yerr=bl_yerr[:, m_ci_bl], fmt="none",
                capsize=2, linewidth=1.0, alpha=0.9)

    ax.bar(x[m_id] + offset[m_id], id_acc_main[m_id],
           width=w[m_id], align="center", alpha=0.85,
           label="IdenGate")

    id_gap = np.clip(conf_ref - id_acc_main, 0.0, 1.0)
    ax.bar(x[m_id] + offset[m_id], id_gap[m_id],
           bottom=id_acc_main[m_id], width=w[m_id], align="center", alpha=0.25)

    ax.errorbar(x[m_ci_id] + offset[m_ci_id], id_acc_mean[m_ci_id],
                yerr=id_yerr[:, m_ci_id], fmt="none",
                capsize=2, linewidth=1.0, alpha=0.9)

    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), visible=False)

    ax2.bar(x - offset, bl_cnt_main, width=w, align="center", alpha=0.75, label="Baseline count")
    ax2.bar(x + offset, id_cnt_main, width=w, align="center", alpha=0.75, label="IdenGate count")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    out_png = OUT_DIR / "Reliability_RetinaMNIST.png"
    out_pdf = OUT_DIR / "Reliability_RetinaMNIST.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("[OK] Done.")
    print(f"[OK] Output directory: {OUT_DIR}")
    print("[OK] Figures:")
    print(f" - {out_png.name}")
    print(f" - {out_pdf.name}")
    print("[OK] CSV:")
    print(f" - {ece_csv.name}")
    print(" - Reliability_bins_baseline_seed*.csv")
    print(" - Reliability_bins_idengate_seed*.csv")
    print("[INFO] Main curve NPZ:")
    print(f" - Baseline: {BASELINE_MAIN}")
    print(f" - IdenGate: {IDENGATE_MAIN}")


if __name__ == "__main__":
    main()
