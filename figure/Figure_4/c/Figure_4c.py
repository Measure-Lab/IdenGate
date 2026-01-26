import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

COLOR_ON_LINE = '#3e9d35'
COLOR_ON_SHADE = '#8cd3a8'
COLOR_OFF_LINE = '#7f7f7f'
COLOR_OFF_SHADE = '#e9e9e9'


save_path = r'file_name'

contrast_factors = np.array([0.800, 0.750, 0.700, 0.650, 0.600])

mgf_off_mean = np.array([0.495, 0.481, 0.453, 0.422, 0.373])
mgf_off_ci = np.array([0.038, 0.042, 0.045, 0.040, 0.055])

mgf_on_mean = np.array([0.510, 0.490, 0.458, 0.432, 0.390])
mgf_on_ci = np.array([0.020, 0.030, 0.050, 0.055, 0.068])

df = pd.DataFrame({
    'Contrast_Factor': contrast_factors,
    'MGF_OFF_Mean': mgf_off_mean,
    'MGF_OFF_95CI': mgf_off_ci,
    'MGF_ON_Mean': mgf_on_mean,
    'MGF_ON_95CI': mgf_on_ci
})

if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_file = os.path.join(save_path, 'results_data.csv')
df.to_csv(csv_file, index=False)
print(f"out CSV: {csv_file}")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

ax.plot(contrast_factors, mgf_off_mean, color=COLOR_OFF_LINE, linestyle='--',
        marker='o', markersize=6, label='MGF OFF', linewidth=2, zorder=2)
ax.fill_between(contrast_factors, mgf_off_mean - mgf_off_ci, mgf_off_mean + mgf_off_ci,
                color=COLOR_OFF_SHADE, alpha=0.7, zorder=1)

ax.plot(contrast_factors, mgf_on_mean, color=COLOR_ON_LINE, linestyle='-',
        marker='s', markersize=6, label='MGF ON', linewidth=2, zorder=4)
ax.fill_between(contrast_factors, mgf_on_mean - mgf_on_ci, mgf_on_mean + mgf_on_ci,
                color=COLOR_ON_SHADE, alpha=0.6, zorder=3)

ax.set_xlim(0.81, 0.59)
ax.set_xticks(contrast_factors)
ax.set_xticklabels(['0.800', '0.750', '0.700', '0.650', '0.600'])

ax.set_xlabel('Contrast Factor (lower = more severe)', fontsize=18, labelpad=12)
ax.set_ylabel(r'Accuracy ($\uparrow$)', fontsize=18, labelpad=12)
ax.set_title(r'Accuracy under low-contrast shift' + '\n' + r'(mean $\pm$ 95% CI)',
             fontsize=20, pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=15)

legend = ax.legend(fontsize=16, loc='lower left', frameon=True)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
png_output = os.path.join(save_path, 'accuracy_plot_updated.png')
pdf_output = os.path.join(save_path, 'accuracy_plot_updated.pdf')

plt.savefig(png_output, bbox_inches='tight', dpi=300)
plt.savefig(pdf_output, bbox_inches='tight')

print(f"output: \nPNG: {png_output} \nPDF: {pdf_output}")
plt.show()