import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.lines import Line2D

data_dir = r"file_name"
csv_path = os.path.join(data_dir, "Figure_4f_data.csv")

if not os.path.exists(csv_path):
    print(f"No file {csv_path}")
else:
    df = pd.read_csv(csv_path)

    alpha = df['Alpha'].tolist()
    mgf_on_060 = df['MGF_ON_c060_mean'].tolist()
    err_060 = df['MGF_ON_c060_std'].tolist()
    mgf_on_080 = df['MGF_ON_c080_mean'].tolist()
    err_080 = df['MGF_ON_c080_std'].tolist()
    baseline_060 = df['MGF_OFF_c060'].iloc[0]
    baseline_080 = df['MGF_OFF_c080'].iloc[0]

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    ax.axhline(y=baseline_060, color='gray', linestyle=(0, (5, 5)), linewidth=1.5, zorder=1)
    ax.axhline(y=baseline_080, color='gray', linestyle='-', linewidth=1.5, zorder=1)

    green_color = '#2ca02c'
    ax.errorbar(alpha, mgf_on_060, yerr=err_060, fmt='--^', color=green_color,
                capsize=3, elinewidth=1.2, markersize=8, zorder=3)
    ax.errorbar(alpha, mgf_on_080, yerr=err_080, fmt='-s', color=green_color,
                capsize=3, elinewidth=1.2, markersize=8, zorder=3)

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle=(0, (3, 3)), lw=1.5, label='MGF OFF (c=0.60)'),
        Line2D([0], [0], color='gray', linestyle='-', lw=1.5, label='MGF OFF (c=0.80)'),
        Line2D([0], [0], color=green_color, linestyle=(0, (3, 3)), marker='^', markersize=8, label='MGF ON (c=0.60)'),
        Line2D([0], [0], color=green_color, linestyle='-', marker='s', markersize=8, label='MGF ON (c=0.80)')
    ]

    ax.legend(handles=legend_elements, fontsize=18, loc='center right',
              frameon=True, edgecolor='lightgray', handlelength=1.8,
              labelspacing=0.4, borderpad=0.6)


    ax.set_title(r'AURC vs $\alpha$ under contrast shift' + '\n' + r'(c$ \in [0.60, 0.80]$; lower = more severe)',
                 fontsize=22, pad=15)
    ax.set_xlabel(r'Conditioning strength $\alpha$', fontsize=26, labelpad=10)
    ax.set_ylabel(r'AURC ($\downarrow$)', fontsize=26, labelpad=10)

    y_ticks = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

    ax.set_xticks(alpha)
    ax.set_xticklabels(['0.0', '0.5', '1.0', '2.0', '4.0'])

    ax.tick_params(axis='both', which='major', labelsize=22, length=6, width=1.2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0.24, 0.63)

    plt.tight_layout()

    plt.savefig(os.path.join(data_dir, "Figure_4f_from_CSV.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, "Figure_4f_from_CSV.png"), bbox_inches='tight')

    print("success")
    plt.show()