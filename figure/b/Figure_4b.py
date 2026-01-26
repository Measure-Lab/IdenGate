import matplotlib.pyplot as plt
import pandas as pd
import os


output_dir = r'file_output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = {
    'Noise_intensity': [0.05, 0.10, 0.15, 0.20, 0.25],
    'MGF_OFF': [0.529, 0.518, 0.514, 0.492, 0.475],
    'MGF_OFF_Lower': [0.494, 0.485, 0.472, 0.453, 0.422],
    'MGF_OFF_Upper': [0.565, 0.552, 0.558, 0.531, 0.529],
    'MGF_ON': [0.520, 0.524, 0.522, 0.514, 0.504],
    'MGF_ON_Lower': [0.502, 0.497, 0.509, 0.492, 0.472],
    'MGF_ON_Upper': [0.538, 0.551, 0.535, 0.535, 0.538]
}
df = pd.DataFrame(data)

df.to_csv(os.path.join(output_dir, 'experiment_data.csv'), index=False)

COLOR_ON_LINE = '#3e9d35'
COLOR_ON_SHADE = '#8cd3a8'
COLOR_OFF_LINE = '#7f7f7f'
COLOR_OFF_SHADE = '#e9e9e9'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

ax.fill_between(df['Noise_intensity'], df['MGF_OFF_Lower'], df['MGF_OFF_Upper'],
                color=COLOR_OFF_SHADE, alpha=0.7, zorder=1)
ax.plot(df['Noise_intensity'], df['MGF_OFF'], color=COLOR_OFF_LINE, linestyle='--',
        marker='o', markersize=8, label='MGF OFF', linewidth=2, zorder=2)

ax.fill_between(df['Noise_intensity'], df['MGF_ON_Lower'], df['MGF_ON_Upper'],
                color=COLOR_ON_SHADE, alpha=0.5, zorder=3)
ax.plot(df['Noise_intensity'], df['MGF_ON'], color=COLOR_ON_LINE, linestyle='-',
        marker='s', markersize=8, label='MGF ON', linewidth=2, zorder=4)

ax.set_title('Accuracy under sensor-noise shift\n(mean ± 95% CI)', fontsize=24, pad=15)
ax.set_xlabel('Noise intensity', fontsize=22, labelpad=10)
ax.set_ylabel('Accuracy (↑)', fontsize=22, labelpad=10)

ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
ax.set_yticks([0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56])
ax.set_xlim(0.04, 0.26)
ax.set_ylim(0.415, 0.575)
ax.tick_params(axis='both', which='major', labelsize=18)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

legend = ax.legend(fontsize=20, loc='lower left', frameon=True, borderpad=0.3)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_edgecolor('#cccccc')

plt.tight_layout()

image_path = os.path.join(output_dir, 'sensor_noise_accuracy_plot.png')
plt.savefig(image_path, bbox_inches='tight')
print(f"save success: {output_dir}")

plt.show()