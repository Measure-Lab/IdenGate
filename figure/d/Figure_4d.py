import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

COLOR_ON_LINE = '#3e9d35'
COLOR_ON_SHADE = '#8cd3a8'
COLOR_OFF_LINE = '#7f7f7f'
COLOR_OFF_SHADE = '#e9e9e9'

save_path = r'file_name'

translation_shifts = np.array([2, 4, 6, 8, 10])

mgf_off_mean = np.array([0.508, 0.478, 0.418, 0.320, 0.194])
mgf_off_ci = np.array([0.040, 0.032, 0.065, 0.130, 0.050])

mgf_on_mean = np.array([0.508, 0.479, 0.446, 0.405, 0.238])
mgf_on_ci = np.array([0.022, 0.025, 0.038, 0.050, 0.070])

df = pd.DataFrame({
    'Translation_Shift_Pixels': translation_shifts,
    'MGF_OFF_Mean': mgf_off_mean,
    'MGF_OFF_95CI': mgf_off_ci,
    'MGF_ON_Mean': mgf_on_mean,
    'MGF_ON_95CI': mgf_on_ci
})

if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_file = os.path.join(save_path, 'results_translation.csv')
df.to_csv(csv_file, index=False)
print(f"out: {csv_file}")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

ax.plot(translation_shifts, mgf_off_mean, color=COLOR_OFF_LINE, linestyle='--',
        marker='o', markersize=6, label='MGF OFF', linewidth=2, zorder=2)
ax.fill_between(translation_shifts, mgf_off_mean - mgf_off_ci, mgf_off_mean + mgf_off_ci,
                color=COLOR_OFF_SHADE, alpha=0.7, zorder=1)

ax.plot(translation_shifts, mgf_on_mean, color=COLOR_ON_LINE, linestyle='-',
        marker='s', markersize=6, label='MGF ON', linewidth=2, zorder=4)
ax.fill_between(translation_shifts, mgf_on_mean - mgf_on_ci, mgf_on_mean + mgf_on_ci,
                color=COLOR_ON_SHADE, alpha=0.6, zorder=3)

ax.set_xlim(1.5, 10.5)
ax.set_xticks(translation_shifts)
ax.set_ylim(0.12, 0.57)

ax.set_xlabel('Translation shift (pixels)', fontsize=18, labelpad=12)
ax.set_ylabel(r'Accuracy ($\uparrow$)', fontsize=18, labelpad=12)
ax.set_title(r'Accuracy under translation shift' + '\n' + r'(mean $\pm$ 95% CI)',
             fontsize=20, pad=20)

# 移除上、右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=15)

legend = ax.legend(fontsize=16, loc='lower left', frameon=True)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()

png_output = os.path.join(save_path, 'accuracy_translation_plot.png')
pdf_output = os.path.join(save_path, 'accuracy_translation_plot.pdf')

plt.savefig(png_output, bbox_inches='tight', dpi=300)
plt.savefig(pdf_output, bbox_inches='tight')

print(f"output: \nPNG: {png_output} \nPDF: {pdf_output}")
plt.show()