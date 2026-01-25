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

X_MIN = 0.00
Y_MIN_MAIN = 0.00
Y_MAX_MAIN = 0.20

def load_conf_correct(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    if "conf" not in d.files:
        raise KeyError(f"'conf' not found in {npz_path}. keys={d.files}")
    conf = d["conf"].astype(np.float64).reshape(-1)

    if "correct" in d.files:
        correct = d["correct"].astype(np.int32).reshape(-1)
    else:
        if ("y_true" not in d.files) or ("y_pred" not in d.files):
            raise KeyError(f"Need 'correct' or ('y_true','y_pred') in {npz_path}. keys={d.files}")
        y_true = d["y_true"].reshape(-1)
        y_pred = d["y_pred"].reshape(-1)
        correct = (y_true == y_pred).astype(np.int32)

    return conf, correct


def compute_risk_coverage(conf: np.ndarray, correct: np.ndarray):
    conf = conf.reshape(-1).astype(np.float64)
    correct = correct.reshape(-1).astype(np.int64)
    n = conf.shape[0]
    if n == 0:
        raise ValueError("Empty input.")

    order = np.argsort(-conf, kind="mergesort")
    correct_sorted = correct[order]

    wrong = 1 - correct_sorted
    cum_wrong = np.cumsum(wrong)
    k = np.arange(1, n + 1, dtype=np.float64)

    coverage = k / float(n)
    risk = cum_wrong / k
    return coverage, risk


def save_curve_csv(path: Path, header, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in zip(*cols):
            w.writerow([f"{float(v):.10f}" for v in row])


def _resample_to_grid(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray):
    return np.interp(x_tgt, x_src, y_src)


def mean_ci_from_runs(curves_cov, curves_risk):
    lengths = [len(c) for c in curves_cov]
    idx_long = int(np.argmax(lengths))
    cov_grid = curves_cov[idx_long].astype(np.float64)

    risk_mat = []
    for cov, risk in zip(curves_cov, curves_risk):
        cov = cov.astype(np.float64)
        risk = risk.astype(np.float64)
        if len(cov) != len(cov_grid) or not np.allclose(cov, cov_grid, atol=0, rtol=0):
            risk = _resample_to_grid(cov, risk, cov_grid)
        risk_mat.append(risk)

    risk_mat = np.stack(risk_mat, axis=0)
    mean = risk_mat.mean(axis=0)

    s = risk_mat.shape[0]
    if s <= 1:
        lo, hi = mean.copy(), mean.copy()
        return cov_grid, mean, lo, hi

    std = risk_mat.std(axis=0, ddof=1)

    if s == 2:
        tcrit = 12.7062047364
    elif s == 3:
        tcrit = 4.3026527299
    elif s == 4:
        tcrit = 3.1824463053
    elif s == 5:
        tcrit = 2.7764451052
    else:
        tcrit = 1.96

    half = tcrit * std / np.sqrt(s)
    lo = mean - half
    hi = mean + half
    return cov_grid, mean, lo, hi


def _mask_from_xmin(x: np.ndarray, xmin: float):
    return x >= xmin


def main():
    conf_i_main, corr_i_main = load_conf_correct(IDENGATE_MAIN)
    conf_b_main, corr_b_main = load_conf_correct(BASELINE_MAIN)

    cov_i_main, risk_i_main = compute_risk_coverage(conf_i_main, corr_i_main)
    cov_b_main, risk_b_main = compute_risk_coverage(conf_b_main, corr_b_main)

    id_covs, id_risks = [], []
    for p in IDENGATE_PATHS:
        conf, corr = load_conf_correct(p)
        cov, risk = compute_risk_coverage(conf, corr)
        id_covs.append(cov)
        id_risks.append(risk)

    bl_covs, bl_risks = [], []
    for p in BASELINE_PATHS:
        conf, corr = load_conf_correct(p)
        cov, risk = compute_risk_coverage(conf, corr)
        bl_covs.append(cov)
        bl_risks.append(risk)

    cov_id_ci, mean_id, lo_id, hi_id = mean_ci_from_runs(id_covs, id_risks)
    cov_bl_ci, mean_bl, lo_bl, hi_bl = mean_ci_from_runs(bl_covs, bl_risks)

    save_curve_csv(
        OUT_DIR / "risk_coverage_idengate_main.csv",
        ["coverage", "risk"],
        [cov_i_main, risk_i_main],
    )
    save_curve_csv(
        OUT_DIR / "risk_coverage_baseline_main.csv",
        ["coverage", "risk"],
        [cov_b_main, risk_b_main],
    )
    save_curve_csv(
        OUT_DIR / "risk_coverage_idengate_mean_ci.csv",
        ["coverage", "risk_mean", "ci_low", "ci_high"],
        [cov_id_ci, mean_id, lo_id, hi_id],
    )
    save_curve_csv(
        OUT_DIR / "risk_coverage_baseline_mean_ci.csv",
        ["coverage", "risk_mean", "ci_low", "ci_high"],
        [cov_bl_ci, mean_bl, lo_bl, hi_bl],
    )

    if len(cov_bl_ci) != len(cov_id_ci) or not np.allclose(cov_bl_ci, cov_id_ci, atol=0, rtol=0):
        mean_id_on_bl = _resample_to_grid(cov_id_ci, mean_id, cov_bl_ci)
        delta_cov = cov_bl_ci
        delta_mean = mean_bl - mean_id_on_bl
    else:
        delta_cov = cov_bl_ci
        delta_mean = mean_bl - mean_id

    m_main_b = _mask_from_xmin(cov_b_main, X_MIN)
    m_main_i = _mask_from_xmin(cov_i_main, X_MIN)
    m_ci_bl = _mask_from_xmin(cov_bl_ci, X_MIN)
    m_ci_id = _mask_from_xmin(cov_id_ci, X_MIN)
    m_delta = _mask_from_xmin(delta_cov, X_MIN)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    ax.fill_between(cov_bl_ci[m_ci_bl], lo_bl[m_ci_bl], hi_bl[m_ci_bl], alpha=0.18)
    ax.plot(cov_b_main[m_main_b], risk_b_main[m_main_b], linewidth=2, label="Baseline")

    ax.fill_between(cov_id_ci[m_ci_id], lo_id[m_ci_id], hi_id[m_ci_id], alpha=0.18)
    ax.plot(cov_i_main[m_main_i], risk_i_main[m_main_i], linewidth=2, label="IdenGate")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Risk–Coverage")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.set_xlim(left=X_MIN, right=1.0)

    ax.set_ylim(bottom=Y_MIN_MAIN, top=Y_MAX_MAIN)

    out_png = OUT_DIR / "RiskCoverage_retina_seed3_baseline_vs_idengate_CI.png"
    out_pdf = OUT_DIR / "RiskCoverage_retina_seed3_baseline_vs_idengate_CI.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    plt.figure(figsize=(6.0, 3.2))
    plt.plot(delta_cov[m_delta], delta_mean[m_delta], linewidth=2)
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Coverage")
    plt.ylabel("ΔRisk (Baseline − IdenGate)")
    plt.title("Risk Difference Curve")
    plt.grid(True, alpha=0.3)
    plt.xlim(left=X_MIN, right=1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "RiskCoverage_delta_mean.png", dpi=300)
    plt.close()

    print("[OK] Done.")
    print(f"[OK] Output directory: {OUT_DIR}")
    print("[OK] Figures:")
    print(" - RiskCoverage_retina_seed3_baseline_vs_idengate_CI.png")
    print(" - RiskCoverage_retina_seed3_baseline_vs_idengate_CI.pdf")
    print(" - RiskCoverage_delta_mean.png")
    print("[OK] CSV:")
    print(" - risk_coverage_baseline_main.csv")
    print(" - risk_coverage_idengate_main.csv")
    print(" - risk_coverage_baseline_mean_ci.csv")
    print(" - risk_coverage_idengate_mean_ci.csv")


if __name__ == "__main__":
    main()
