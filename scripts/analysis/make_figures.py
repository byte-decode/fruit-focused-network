import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK})
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures") + os.sep
os.makedirs(OUT, exist_ok=True)

def style(ax):
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(ax.get_xlim()[1], 0.505, "chance", ha="right", va="bottom", color=MUTED, fontsize=8)

def bars(ax, x, vals, w, color, err=None, label=None):
    for i, (xi, v) in enumerate(zip(x, vals)):
        if v is None:
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", color=MUTED, fontsize=8)
            continue
        ax.bar(xi, v, w - 0.03, bottom=0, color=color, edgecolor="white", lw=0,
               label=label if i == 0 else None)
        if err is not None and err[i]:
            ax.errorbar(xi, v, yerr=err[i], color=INK, lw=1, capsize=2.5, capthick=1)
        top = v + (err[i] if err is not None and err[i] else 0)
        ax.text(xi, top + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)

# ---- Figure: shortcut audit (honest group split)
cats = ["Full photo", "Fruit only\n(background grey)", "Background only\n(fruit removed)",
        "Border colour\n(24 numbers)", "Fruit size and\nposition (5 numbers)"]
fn = [0.996, 0.970, 0.907, 0.849, 0.578]
fr = [0.999, 0.995, 0.755, None, 0.625]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
x = np.arange(len(cats)); w = 0.38
bars(ax, x - w / 2, fn, w, BLUE, label="FruitNet")
bars(ax, x + w / 2, fr, w, ORANGE, label="Fresh-Rotten")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("Balanced accuracy")
style(ax)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1, 1.12))
fig.tight_layout(); fig.savefig(OUT + "shortcut_audit.pdf"); fig.savefig(OUT + "shortcut_audit.png", dpi=200)

# ---- Figure: FFN vs baseline
cats = ["FruitNet, unseen fruit\n(natural photos)", "FruitNet, unseen fruit\n(background swapped)",
        "Fresh-Rotten real photos\n(trained on FruitNet)"]
base, base_e = [0.861, 0.660, 0.839], [0.016, 0.039, 0.035]
ffn, ffn_e = [0.856, 0.800, 0.912], [0.015, 0.024, 0.004]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
x = np.arange(len(cats)); w = 0.36
bars(ax, x - w / 2, base, w, BLUE, base_e, "Baseline MobileNetV3")
bars(ax, x + w / 2, ffn, w, ORANGE, ffn_e, "FFN (ours)")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("Balanced accuracy (mean ± SD, 3 seeds)")
style(ax)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8, bbox_to_anchor=(1, 1.12))
fig.tight_layout(); fig.savefig(OUT + "ffn_results.pdf"); fig.savefig(OUT + "ffn_results.png", dpi=200)
print("ok")
