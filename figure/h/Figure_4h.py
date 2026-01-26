import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42

# 用户指定颜色
COLOR_ON_LINE = '#3e9d35'
COLOR_ON_SHADE = '#8cd3a8'
COLOR_OFF_LINE = '#7f7f7f'
COLOR_OFF_SHADE = '#e9e9e9'


def plot_perfect_nmi_figure():

    base_path = r'file_name'

    b_main = pd.read_csv(os.path.join(base_path, 'risk_coverage_baseline_main.csv'))
    b_ci = pd.read_csv(os.path.join(base_path, 'risk_coverage_baseline_mean_ci.csv'))
    i_main = pd.read_csv(os.path.join(base_path, 'risk_coverage_idengate_main.csv'))
    i_ci = pd.read_csv(os.path.join(base_path, 'risk_coverage_idengate_mean_ci.csv'))

    b_main = b_main[b_main['coverage'] >= 0.1]
    b_ci = b_ci[b_ci['coverage'] >= 0.1]
    i_main = i_main[i_main['coverage'] >= 0.1]
    i_ci = i_ci[i_ci['coverage'] >= 0.1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.fill_between(b_ci['coverage'], b_ci['ci_low'], b_ci['ci_high'],
                    color=COLOR_OFF_SHADE, alpha=0.7, lw=0, label='_nolegend_', zorder=1)

    ax.fill_between(i_ci['coverage'], i_ci['ci_low'], i_ci['ci_high'],
                    color=COLOR_ON_SHADE, alpha=0.6, lw=0, label='_nolegend_', zorder=2)

    ax.plot(b_main['coverage'], b_main['risk'], color=COLOR_OFF_LINE,
            label='MGF OFF', linewidth=1.5, zorder=3)

    ax.plot(i_main['coverage'], i_main['risk'], color=COLOR_ON_LINE,
            label='MGF ON', linewidth=1.5, zorder=4)

    ax.set_title('Risk-Coverage', fontsize=24, pad=12)
    ax.set_xlabel('Coverage', fontsize=22)
    ax.set_ylabel(r'Risk ($\downarrow$)', fontsize=22)

    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.10, 0.46)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])

    ax.tick_params(axis='both', which='major', labelsize=18, direction='out', length=6, width=1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(loc='upper left', fontsize=18, frameon=True, edgecolor='lightgray')
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()

    save_file = os.path.join(base_path, 'h.png')
    plt.savefig(save_file, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_perfect_nmi_figure()