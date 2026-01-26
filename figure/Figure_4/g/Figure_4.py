import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.legend_handler import HandlerErrorbar

output_path = r'file_name'
file_name_img = 'Figure_4.png'
file_name_csv = 'Figure_4_data.csv'

csv_load_path = os.path.join(output_path, file_name_csv)

if not os.path.exists(csv_load_path):
    print(f"No file {csv_load_path} ")
else:
    df = pd.read_csv(csv_load_path)

    alpha = df['alpha'].values
    mgf_on_10_y = df['MGF_ON_t10_mean'].values
    mgf_on_10_err = df['MGF_ON_t10_std'].values
    mgf_on_2_y = df['MGF_ON_t2_mean'].values
    mgf_on_2_err = df['MGF_ON_t2_std'].values

    baseline_10 = df['Baseline_OFF_t10'].iloc[0]
    baseline_2 = df['Baseline_OFF_t2'].iloc[0]


    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.axhline(y=baseline_10, color='gray', linestyle=(0, (5, 5)), linewidth=1.2, label='MGF OFF (t=10px)')
    ax.axhline(y=baseline_2, color='gray', linestyle='-', linewidth=1.2, label='MGF OFF (t=2px)')

    line1 = ax.errorbar(alpha, mgf_on_10_y, yerr=mgf_on_10_err, fmt='--^', color='#2ca02c',
                        capsize=3, markersize=8, linewidth=1.5, label='MGF ON (t=10px)',
                        elinewidth=1.2, capthick=1.2)

    line2 = ax.errorbar(alpha, mgf_on_2_y, yerr=mgf_on_2_err, fmt='-s', color='#2ca02c',
                        capsize=3, markersize=8, linewidth=1.5, label='MGF ON (t=2px)',
                        elinewidth=1.2, capthick=1.2)

    ax.set_title(r'AURC vs $\alpha$ under translation shift' + '\n' + r'(t$ \in [2, 10]$ px; higher = more severe)',
                 fontsize=20, pad=15, fontweight='normal')
    ax.set_xlabel(r'Conditioning strength $\alpha$', fontsize=22)
    ax.set_ylabel(r'AURC ($\downarrow$)', fontsize=22)

    ax.set_xticks(alpha)
    ax.set_xticklabels(['0.0', '0.5', '1.0', '2.0', '4.0'], fontsize=18)
    ax.set_yticks(np.arange(0.28, 0.42, 0.02))
    ax.set_yticklabels([f'{x:.2f}' for x in np.arange(0.28, 0.42, 0.02)], fontsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(width=1.2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)

    ax.legend(fontsize=15, frameon=True, loc='upper right', edgecolor='lightgray',
              handler_map={type(line1): HandlerErrorbar(xerr_size=0, yerr_size=0)})

    plt.tight_layout()

    img_save_path = os.path.join(output_path, file_name_img)
    plt.savefig(img_save_path, bbox_inches='tight', dpi=300)
    print(f"out: {img_save_path}")

    plt.show()