import matplotlib
matplotlib.use('Agg')  # use non-interactive backend, no Tk needed
import matplotlib.pyplot as plt
import json

with open("results_exp1.json") as f:
    data = json.load(f)

erm = data["history_erm"]
mixup = data["history_mixup"]
epochs = range(1, len(erm) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, erm, label="ERM", linewidth=1.5, alpha=0.8)
plt.plot(epochs, mixup, label="mixup (α=1)", linewidth=1.5, alpha=0.8)

# red line at epoch 100 (learning rate step)
"""
comment for no red line
"""
plt.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
plt.text(102, plt.ylim()[0] + 1, 'lr step', color='red', fontsize=9, alpha=0.8)

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("CIFAR-10: ERM vs mixup (PreAct ResNet-18, 200 epochs)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("exp1_accuracy_with_red.png", dpi=200)
plt.close()