import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 48,
        "axes.labelsize": 40,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 28,
        "figure.titlesize": 11,
    }
)
dimensions = [1024, 2048, 4096, 8192, 16384, 32768]
int8_non_causal = [142.7, 169.29, 199.5, 201.59, 200.47, 201.16]
int4_non_causal = [183.43, 196.81, 197.6, 199.84, 200.01, 202.3]
fp16_non_causal = [64.24, 74.73, 80.06, 83.23, 85.9, 86.78]
int8_causal = [115.66, 147.7, 167.77, 179.22, 185.56, 188.76]
int4_causal = [157.13, 176.5, 189.89, 197.03, 199.01, 199.29]
fp16_causal = [65.7, 69.75, 85.48, 86.33, 87.91, 87.51]
fig, axs = plt.subplots(2, 1, figsize=(16, 12), dpi=120)
ax = axs[0]
ax.plot(
    dimensions, int8_non_causal, marker="o", markersize=5, linewidth=1.5, label="INT8"
)
ax.plot(
    dimensions, int4_non_causal, marker="s", markersize=5, linewidth=1.5, label="INT4"
)
ax.plot(
    dimensions, fp16_non_causal, marker="^", markersize=5, linewidth=1.5, label="FP16"
)
ax.set_title("Non-causal Attention Performance", pad=10)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xticks(dimensions)
ax.set_ylim(50, 220)
ax = axs[1]
ax.plot(dimensions, int8_causal, marker="o", markersize=5, linewidth=1.5, label="INT8")
ax.plot(dimensions, int4_causal, marker="s", markersize=5, linewidth=1.5, label="INT4")
ax.plot(dimensions, fp16_causal, marker="^", markersize=5, linewidth=1.5, label="FP16")
ax.set_title("Causal Attention Performance", pad=10)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend(framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xticks(dimensions)
ax.set_ylim(50, 220)
plt.tight_layout(pad=2.0, h_pad=2.5)
plt.savefig("attention_performance.png", bbox_inches="tight", dpi=300)
plt.savefig("attention_performance.svg", bbox_inches="tight", dpi=300)
plt.savefig("attention_performance.pdf", bbox_inches="tight", dpi=300)
print("Finished generating attention_performance.png and .pdf")
plt.close()
