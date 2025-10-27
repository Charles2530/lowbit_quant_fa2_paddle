import matplotlib.pyplot as plt
import numpy as np

dimensions = [1024, 2048, 4096, 8192]
int8_non_causal = [142.7, 169.29, 199.5, 201.59]
int4_non_causal = [83.43, 96.81, 97.6, 99.84]
fp16_non_causal = [133.7, 148.75, 179.48, 180.33]
int8_causal = [115.66, 147.7, 167.77, 179.22]
int4_causal = [57.13, 76.5, 89.89, 97.03]
fp16_causal = [118.24, 150.73, 164.06, 172.23]
mixed_precision_non_causal = {
    (0): [91.33, 122.18, 156.09, 167.74],
    (30): [110.46, 146.8, 162.98, 167.69],
    (50): [110.6, 148.62, 162.72, 167.75],
    (70): [110.33, 146.98, 162.95, 167.73],
    (100): [110.31, 148.75, 162.81, 167.71],
}
mixed_precision_causal = {
    (0): [55.08, 74.28, 81.45, 83.88],
    (30): [55.15, 74.3, 81.47, 83.86],
    (50): [55.15, 74.37, 81.48, 83.88],
    (70): [55.17, 74.37, 81.4, 83.87],
    (100): [55.12, 74.3, 81.47, 83.86],
}
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
ax = axs[0, 0]
ax.plot(dimensions, int8_non_causal, marker="o", label="INT8")
ax.plot(dimensions, int4_non_causal, marker="s", label="INT4")
ax.plot(dimensions, fp16_non_causal, marker="^", label="FP16")
ax.set_title(
    "Non-causal Attention Performance Comparison for Different Quantization Bitwidths"
)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(dimensions)
ax = axs[0, 1]
ax.plot(dimensions, int8_causal, marker="o", label="INT8")
ax.plot(dimensions, int4_causal, marker="s", label="INT4")
ax.plot(dimensions, fp16_causal, marker="^", label="FP16")
ax.set_title(
    "Causal Attention Performance Comparison for Different Quantization Bitwidths"
)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(dimensions)
ax = axs[1, 0]
for ratio, values in mixed_precision_non_causal.items():
    ax.plot(dimensions, values, marker="o", label=f"{ratio}%")
ax.set_title(
    "Non-causal Attention Performance Comparison for Mixed Precision Quantization Strategy"
)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(dimensions)
ax = axs[1, 1]
for ratio, values in mixed_precision_causal.items():
    ax.plot(dimensions, values, marker="o", label=f"{ratio}%")
ax.set_title(
    "Causal Attention Performance Comparison for Mixed Precision Quantization Strategy"
)
ax.set_xlabel("Matrix Dimension")
ax.set_ylabel("TFLOP/s")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_xticks(dimensions)
plt.tight_layout()
plt.savefig("attention_performance.png")
plt.savefig("attention_performance.pdf")
plt.close()
