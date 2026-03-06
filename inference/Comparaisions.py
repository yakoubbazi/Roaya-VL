import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Data (edit if needed)
# ----------------------------
models = [
    "InternVL3.5-2B",
    "Qwen2.5-VL-3B",
    "InternVL3-2B",
    "LFM2-VL-3B",
    "SmolVLM2-2.2B",
    "Roa'ya (45.5K)",
]

data = {
    "MMBench":     [78.18, 80.41, 81.10, 79.81, 69.24, 69.61],
    "MMStar":      [57.67, 56.13, 61.10, 57.73, 46.00, 50.00],
    "MMMU (val)":  [51.78, 51.67, 48.70, 45.33, 41.60, 40.44],
    "POPE":        [87.17, 86.17, 90.10, 89.01, 85.10, 82.46],
    "RealWorldQA": [60.78, 65.23, 65.10, 71.37, 57.50, 59.38],
    "BLINK":       [50.97, 48.97, 53.10, 51.03, 42.30, 43.81],
    # NOTE: you had a typo/garble in InfoVQA for InternVL3-2B ("650.6.10").
    # I set it to NaN; replace with the correct value.
    "InfoVQA":     [69.29, 76.12, np.nan, 67.37, 37.75, 40.55],
    # OCRBench scale differs across papers; still plotted as you provided
    "OCRBench":    [834, 824, 831, 822, 725, 595],
    "SEEDBench":   [75.41, 73.88, 74.95, 76.55, 71.30, 71.17],
    "AI2D":        [81.50, np.nan, np.nan, np.nan, np.nan, 71.53],
    "MME":         [2128, 2163, 2186, 2050, 1792, np.nan],
    "MathVista":   [61.60, 62.50, 57.60, 62.20, 51.50, np.nan],
}

# ----------------------------
# 2) Layout like your example
# ----------------------------
benchmarks = list(data.keys())

# choose grid shape (e.g., 3 columns)
ncols = 3
nrows = int(np.ceil(len(benchmarks) / ncols))

fig_w = 16
fig_h = 4.4 * nrows
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
axes = np.array(axes).reshape(-1)

x = np.arange(len(models))
bar_w = 0.8

# ----------------------------
# 3) Draw bars
# ----------------------------
for i, b in enumerate(benchmarks):
    ax = axes[i]
    vals = np.array(data[b], dtype=float)

    # mask NaNs so missing values don't draw
    mask = ~np.isnan(vals)
    ax.bar(x[mask], vals[mask], width=bar_w)

    # value labels
    for xi, yi in zip(x[mask], vals[mask]):
        ax.text(xi, yi, f"{yi:.2f}" if yi < 1000 else f"{int(yi)}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(b, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

# hide unused subplots
for j in range(len(benchmarks), len(axes)):
    axes[j].axis("off")

# ----------------------------
# 4) Overall title + styling
# ----------------------------
fig.suptitle("Model Performance Evaluation (Your Results)", fontsize=18, fontweight="bold", y=0.995)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.show()
