import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

with open("results_exp2.json") as f:
    data = json.load(f)

alphas = []
means = []
stds = []
for key, val in data.items():
    alphas.append(val["alpha"])
    means.append(val["last10_mean"])
    stds.append(val["last10_std"])

# sort by alpha
order = np.argsort(alphas)
alphas = np.array(alphas)[order]
means = np.array(means)[order]
stds = np.array(stds)[order]

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
x = np.arange(len(alphas))

plt.figure(figsize=(6, 5))
bars = plt.bar(x, means, yerr=stds, color=colors, capsize=6, edgecolor='black', linewidth=0.8)
plt.xticks(x, [f'α={a}' for a in alphas])
plt.ylabel('Test Accuracy (%)')
plt.title('mixup: ablation over α (50 epochs)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# annotate bars with values
for bar, m in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{m:.2f}%', ha='center', va='bottom', fontsize=9)

plt.ylim(85, max(means) + max(stds) + 2)
plt.tight_layout()
plt.savefig("exp2_ablation.png", dpi=200)
plt.close()