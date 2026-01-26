import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def wilson_ci(p_pct, n, z=1.96):
    p = p_pct / 100.0
    denominator = 1 + z ** 2 / n
    centre_adjusted_probability = p + z ** 2 / (2 * n)
    adjusted_variance = np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    lower = (centre_adjusted_probability - z * adjusted_variance) / denominator
    upper = (centre_adjusted_probability + z * adjusted_variance) / denominator
    return lower * 100, upper * 100

FILE_DIR = r'file_name'
CSV_NAME = 'BloodMNIST.csv'
CSV_PATH = os.path.join(FILE_DIR, CSV_NAME)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"No CSV: {CSV_PATH}")
        return

    file_base_name = os.path.splitext(CSV_NAME)[0]
    save_name = f"{file_base_name}_accuracy_plot.png"
    save_path = os.path.join(FILE_DIR, save_name)

    df = pd.read_csv(CSV_PATH)
    df = df.iloc[::-1].reset_index(drop=True)

    dynamic_n = df['n_samples'].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, row in df.iterrows():
        acc = row['accuracy']
        ci_l, ci_h = row['ci_lower'], row['ci_upper']
        name = row['model_name']
        hl = row['highlight']

        p_val = acc / 100.0
        se = np.sqrt(p_val * (1 - p_val) / dynamic_n) * 100
        box_width_offset = se * 0.75

        if hl == 'green' or "Ours" in str(name):
            ax.axhspan(i - 0.5, i + 0.5, xmin=-0.3, xmax=1, color='#d9ead3', zorder=0, clip_on=False)
        elif hl == 'blue':
            ax.axhspan(i - 0.5, i + 0.5, xmin=-0.3, xmax=1, color='#e1f5fe', zorder=0, clip_on=False)

        ax.hlines(i, ci_l, ci_h, colors='black', linewidth=1, zorder=2)
        ax.vlines([ci_l, ci_h], i - 0.1, i + 0.1, colors='black', linewidth=1, zorder=2)

        rect = plt.Rectangle((acc - box_width_offset, i - 0.25),
                             2 * box_width_offset, 0.5,
                             facecolor='#81d4fa', edgecolor='black', linewidth=0.8, zorder=3)
        ax.add_patch(rect)

        ax.vlines(acc, i - 0.25, i + 0.25, colors='black', linewidth=1, zorder=4)
        ax.text(acc, i + 0.3, f'{acc:.1f}', ha='center', va='bottom', fontsize=16, fontname='Arial')

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model_name'], fontsize=20, fontname='Arial')

    ax.set_xlabel('Top-1 accuracy (%)with 95% Wilson CI', fontsize=20, labelpad=15, fontname='Arial')

    axis_map = {
        "Blood": (86, 98, 2), "Breast": (60, 95, 5), "Derma": (66, 80, 2),
        "OCT": (55, 85, 5), "Path": (70, 95, 5), "Pneumonia": (75, 95, 5),
        "Retina": (45, 80, 5), "Tissue": (52, 72, 2)
    }

    start_x, end_x, step = 0, 100, 5
    for key, val in axis_map.items():
        if key in file_base_name:
            start_x, end_x, step = val
            break

    ax.set_xticks(np.arange(start_x, end_x + 1, step))
    ax.set_xlim(start_x - (step * 0.4), end_x + step)

    ours_row = df[df['model_name'].str.contains("Ours")]
    ours_acc = ours_row['accuracy'].iloc[0] if not ours_row.empty else df['accuracy'].iloc[-1]
    baseline_accs = df[~df['model_name'].str.contains("Ours")]['accuracy']
    best_baseline = baseline_accs.max() if not baseline_accs.empty else 0

    ax.text(0, 1.08, f'{file_base_name}: Model accuracy with 95% CI',
            transform=ax.transAxes, fontsize=25, fontname='Arial')
    ax.text(0, 1.01,
            f'(n={dynamic_n:,})   $\Delta^*$ = +{ours_acc - best_baseline:.1f} pp (vs best baseline ({best_baseline:.1f}))',
            transform=ax.transAxes, fontsize=20, fontname='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=16, length=8)

    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.savefig(save_path, dpi=300)
    print(f"ok")
    print(f"output: {save_name}")
    plt.show()


if __name__ == "__main__":
    main()
