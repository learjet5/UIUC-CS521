import matplotlib.pyplot as plt
import numpy as np

models = ["pretr_Linf.pth", "pretr_L2.pth", "pretr_RAMP.pth"]

standard_acc = [0.82800, 0.88750, 0.81190]
robust_acc_linf = [0.47050, 0.30850, 0.49760]

x = np.arange(len(models)) 
width = 0.25  # the width of the bars


fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width, standard_acc, width, label="Standard Accuracy")
rects2 = ax.bar(x, robust_acc_linf, width, label="Robust Accuracy (L∞ & L2)")

ax.set_ylabel("Accuracy")
ax.set_xlabel("Models")
ax.set_title("Multi-Norm Robustness Analysis")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height*100:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()