import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

save_path = r"file_name"
csv_file = os.path.join(save_path, "accuracy_degradation_data.csv")

df = pd.read_csv(csv_file)
categories = df['Category'].tolist()
params = [
    p.replace('sigma', '$\\sigma$').replace('->', '$\\to$')
    for p in df['Parameter_Range'].tolist()
]
mgf_off = df['MGF_OFF_pp'].tolist()
mgf_on = df['MGF_ON_pp'].tolist()
reductions = df['Reduction'].tolist()
plt.rcParams['font.family'] = 'Arial'
x_labels = [f"{c}\n{p}" for c, p in zip(categories, params)]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 7))

rects1 = ax.bar(x - width / 2, mgf_off, width, label='MGF OFF', color='#969696')
rects2 = ax.bar(x + width / 2, mgf_on, width, label='MGF ON', color='#2ecc71')

ax.set_ylabel('Accuracy drop $\Delta$ (pp) ($\downarrow$)', fontsize=20)
ax.set_title('Severity-induced accuracy degradation\n(min$\\to$max)', fontsize=20, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=16)
ax.set_ylim(0, 40)

ax.tick_params(axis='y', labelsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i in range(len(categories)):
    ax.text(x[i] - width / 2, mgf_off[i] + 0.5, f'{mgf_off[i]:.1f}pp', ha='center', va='bottom', fontsize=16)
    ax.text(x[i] + width / 2, mgf_on[i] + 0.5, f'{mgf_on[i]:.1f}pp', ha='center', va='bottom', fontsize=16)

    h_line = max(mgf_off[i], mgf_on[i]) + 3.5
    ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
            [mgf_off[i] + 0.5, h_line, h_line, mgf_on[i] + 0.5], color='gray', lw=0.8)

    ax.text(x[i], h_line + 0.5, f'Reduction: {reductions[i]}', ha='center', va='bottom', fontsize=16)

ax.legend(frameon=True, loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=16)
plt.tight_layout()

pdf_file = os.path.join(save_path, "accuracy_degradation.pdf")
png_file = os.path.join(save_path, "accuracy_degradation.png")

plt.savefig(pdf_file, bbox_inches='tight')
plt.savefig(png_file, dpi=300, bbox_inches='tight')

print(f"Reading from: {csv_file}")
print(f"Output: {save_path}")
plt.show()