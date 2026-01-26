import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.0


def plot_figure_4e_from_csv():
    base_dir = r'file_name'
    csv_path = os.path.join(base_dir, 'Figure_4e_data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: No file {csv_path}，。")
        return

    df = pd.read_csv(csv_path)

    alpha = df['alpha'].values
    mgf_on_025 = df['MGF_ON_sigma_0.25_mean'].values
    err_on_025 = df['MGF_ON_sigma_0.25_std'].values
    mgf_on_005 = df['MGF_ON_sigma_0.05_mean'].values
    err_on_005 = df['MGF_ON_sigma_0.05_std'].values
    baseline_high = df['Baseline_OFF_sigma_0.25'].iloc[0]
    baseline_low = df['Baseline_OFF_sigma_0.05'].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    green_color = '#2ca02c'
    gray_color = '#707070'

    ax.axhline(y=baseline_high, color=gray_color, linestyle='--', linewidth=1.2, zorder=1)
    ax.axhline(y=baseline_low, color=gray_color, linestyle='-', linewidth=1.2, zorder=1)

    ax.errorbar(alpha, mgf_on_025, yerr=err_on_025, fmt='--^', color=green_color,
                capsize=3, markersize=8, linewidth=1.8, elinewidth=1.2,
                markerfacecolor=green_color, markeredgecolor=green_color)

    ax.errorbar(alpha, mgf_on_005, yerr=err_on_005, fmt='-s', color=green_color,
                capsize=3, markersize=8, linewidth=1.8, elinewidth=1.2,
                markerfacecolor=green_color, markeredgecolor=green_color)

    ax.set_xlabel('Conditioning strength $\\alpha$', fontsize=18, labelpad=8)
    ax.set_ylabel('AURC ($\downarrow$)', fontsize=18, labelpad=8)
    ax.set_title('AURC vs $\\alpha$ under Gaussian noise\n(min $\sigma = 0.05$, max $\sigma = 0.25$)',
                 fontsize=18, pad=15)

    ax.set_xticks(alpha)
    ax.set_xticklabels([f'{a:.1f}' for a in alpha], fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [
        Line2D([0], [0], color=gray_color, linestyle='--', linewidth=1.2),
        Line2D([0], [0], color=gray_color, linestyle='-', linewidth=1.2),
        Line2D([0], [0], color=green_color, linestyle='--', marker='^', markersize=8, linewidth=1.8),
        Line2D([0], [0], color=green_color, linestyle='-', marker='s', markersize=8, linewidth=1.8)
    ]
    labels = [
        r'MGF OFF ($\sigma$=0.25)',
        r'MGF OFF ($\sigma$=0.05)',
        r'MGF ON ($\sigma$=0.25)',
        r'MGF ON ($\sigma$=0.05)'
    ]

    legend = ax.legend(handles, labels, fontsize=14, loc='center right',
                       frameon=True, framealpha=1,
                       handlelength=2.0, handleheight=0.1, labelspacing=0.5)
    # -----------------------------------------------------------------

    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor('#d3d3d3')

    plt.tight_layout()

    output_file = os.path.join(base_dir, 'Figure_4e.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight', transparent=True)
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)

    plt.show()
    print(f"Success: Figure recreated from CSV and saved to {output_file}")


if __name__ == "__main__":
    plot_figure_4e_from_csv()