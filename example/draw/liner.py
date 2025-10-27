import matplotlib.pyplot as plt
import numpy as np

non_causal_data = {
    "dimensions": [1024, 2048, 4096, 8192],
    "INT8": [142.7, 169.29, 199.5, 201.59],
    "INT4": [183.43, 196.81, 197.6, 199.84],
    "FP16": [133.7, 148.75, 179.48, 180.33],
}
causal_data = {
    "dimensions": [1024, 2048, 4096, 8192],
    "INT8": [115.66, 147.7, 167.77, 179.22],
    "INT4": [157.13, 176.5, 189.89, 197.03],
    "FP16": [118.24, 150.73, 164.06, 172.23],
}
plt.style.use("_mpl-gallery-nogrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(
    non_causal_data["dimensions"],
    non_causal_data["INT8"],
    marker="H",
    label="INT8 TFLOP/s",
    color="#1f77b4",
    linewidth=2,
)
ax1.plot(
    non_causal_data["dimensions"],
    non_causal_data["INT4"],
    marker="H",
    label="INT4 TFLOP/s",
    color="#ff7f0e",
    linewidth=2,
)
ax1.plot(
    non_causal_data["dimensions"],
    non_causal_data["FP16"],
    marker="H",
    label="FP16 TFLOP/s",
    color="#2ca02c",
    linewidth=2,
)
ax1.set_xlabel("Matrix Dimension", fontsize=12, fontfamily="sans-serif")
ax1.set_ylabel("TFLOP/s", fontsize=12, fontfamily="sans-serif")
ax1.set_title(
    "Non-Causal Attention Performance Comparison", fontsize=14, fontfamily="sans-serif"
)
ax1.legend(loc="upper right")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax2.plot(
    causal_data["dimensions"],
    causal_data["INT8"],
    marker="H",
    label="INT8 TFLOP/s",
    color="#1f77b4",
    linewidth=2,
)
ax2.plot(
    causal_data["dimensions"],
    causal_data["INT4"],
    marker="H",
    label="INT4 TFLOP/s",
    color="#ff7f0e",
    linewidth=2,
)
ax2.plot(
    causal_data["dimensions"],
    causal_data["FP16"],
    marker="H",
    label="FP16 TFLOP/s",
    color="#2ca02c",
    linewidth=2,
)
ax2.set_xlabel("Matrix Dimension", fontsize=12, fontfamily="sans-serif")
ax2.set_ylabel("TFLOP/s", fontsize=12, fontfamily="sans-serif")
ax2.set_title(
    "Causal Attention Performance Comparison", fontsize=14, fontfamily="sans-serif"
)
ax2.legend(loc="upper right")
ax2.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("performance_comparison.png", format="png", dpi=300)
plt.savefig("performance_comparison.pdf", format="pdf")
plt.close()
