import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IDENGATE_PATHS = [
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/1/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/2/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/3/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/4/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/42/translation_robustness.csv"),
]
BASELINE_PATHS = [
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/1/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/2/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/3/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/4/translation_robustness.csv"),
    Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/42/translation_robustness.csv"),
]

IDENGATE_MAIN = Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/Idengate/42/translation_robustness.csv")
BASELINE_MAIN = Path("/home/ubuntu/PycharmProjects/MIA/outputs/extral_experiments/baseline/4/translation_robustness.csv")

OUT_DIR = Path("/home/ubuntu/PycharmProjects/MIA/middle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUC_COL = "auc_macro"

def read_curve(csv_path: Path, auc_col: str):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "translation_shift_pixels" not in df.columns:
        raise ValueError(f"'translation_shift_pixels' not found in {csv_path}. Columns={df.columns.tolist()}")
    if auc_col not in df.columns:
        raise ValueError(f"'{auc_col}' not found in {csv_path}. Columns={df.columns.tolist()}")

    x = df["translation_shift_pixels"].astype(float).to_numpy()
    y = df[auc_col].astype(float).to_numpy()
    order = np.argsort(x)
    return x[order], y[order]


def stack_by_x(paths, auc_col: str):
    xs, ys = [], []
    for p in paths:
        x, y = read_curve(p, auc_col)
        xs.append(x)
        ys.append(y)

    x_ref = xs[0]
    Y = []
    for x, y in zip(xs, ys):
        if len(x) == len(x_ref) and np.allclose(x, x_ref):
            Y.append(y)
        else:
            Y.append(np.interp(x_ref, x, y))
    return x_ref, np.vstack(Y)


def mean_ci95(Y):
    n = Y.shape[0]
    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0, ddof=1)
    se = std / math.sqrt(n)
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se

    mean = np.clip(mean, 0.0, 1.0)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    return mean, lo, hi


def interp_if_needed(x_src, y_src, x_tgt):
    if len(x_src) == len(x_tgt) and np.allclose(x_src, x_tgt):
        return y_src
    return np.interp(x_tgt, x_src, y_src)


def main():
    x_id, Y_id = stack_by_x(IDENGATE_PATHS, AUC_COL)
    x_bs, Y_bs = stack_by_x(BASELINE_PATHS, AUC_COL)

    id_mean, id_lo, id_hi = mean_ci95(Y_id)
    bs_mean, bs_lo, bs_hi = mean_ci95(Y_bs)

    x_id_m, y_id_m = read_curve(IDENGATE_MAIN, AUC_COL)
    x_bs_m, y_bs_m = read_curve(BASELINE_MAIN, AUC_COL)

    y_id_m = interp_if_needed(x_id_m, y_id_m, x_id)
    y_bs_m = interp_if_needed(x_bs_m, y_bs_m, x_bs)

    plt.figure(figsize=(7.2, 4.8))

    plt.fill_between(x_bs, bs_lo, bs_hi, alpha=0.20)
    plt.fill_between(x_id, id_lo, id_hi, alpha=0.20)

    plt.plot(x_bs, y_bs_m, linewidth=2.0, label="Baseline")
    plt.plot(x_id, y_id_m, linewidth=2.0, label="IdenGate")

    plt.xlabel("Translatio shift(pixels)")
    plt.ylabel("AUC")
    plt.title("AUC under translation shift")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_png = OUT_DIR / f"SensorNoise_AUC_{AUC_COL}.png"
    out_pdf = OUT_DIR / f"SensorNoise_AUC_{AUC_COL}.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    id_df = pd.DataFrame({
        "translation_shift_pixels": x_id,
        "idengate_mean_auc": id_mean,
        "idengate_ci_low": id_lo,
        "idengate_ci_high": id_hi,
    })
    bs_df = pd.DataFrame({
        "translation_shift_pixels": x_bs,
        "baseline_mean_auc": bs_mean,
        "baseline_ci_low": bs_lo,
        "baseline_ci_high": bs_hi,
    })
    merged = pd.merge(id_df, bs_df, on="translation_shift_pixels", how="outer").sort_values("translation_shift_pixels")
    out_csv = OUT_DIR / f"SensorNoise_AUC_{AUC_COL}_summary.csv"
    merged.to_csv(out_csv, index=False)

    print("[DONE] Saved:")
    print(" -", out_png)
    print(" -", out_pdf)
    print(" -", out_csv)


if __name__ == "__main__":
    main()
