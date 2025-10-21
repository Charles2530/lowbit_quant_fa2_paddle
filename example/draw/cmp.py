import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False
fontSize = 24
fig = plt.figure(figsize=(24, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
methods1 = ["FlashAttention2", "Ours(Mixed Precision)"]
speeds1 = [86, 201]
x_pos1 = np.arange(len(methods1))
bar_width1 = 0.8
bars11 = ax1.bar(0, speeds1[0], bar_width1, color="tomato", label="FlashAttention2")
bars12 = ax1.bar(
    1 + bar_width1,
    speeds1[1],
    bar_width1,
    color="mediumaquamarine",
    label="Ours(Mixed Precision)",
)
ax1.text(
    x_pos1[0],
    speeds1[0] + 0.5,
    str(speeds1[0]),
    ha="center",
    va="bottom",
    fontsize=fontSize * 1.4,
    color="black",
)
ax1.text(
    x_pos1[1] + bar_width1,
    speeds1[1] + 0.5,
    str(speeds1[1]),
    ha="center",
    va="bottom",
    fontsize=fontSize * 1.4,
    color="black",
)
ax1.annotate(
    "2.4x",
    xytext=(0.1, 96),
    xy=(1.3, 191),
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=fontSize * 1.5,
    color="red",
)
ax1.legend(loc="upper left", fontsize=fontSize * 1.8)
ax1.set_ylabel("Kernel Speed (TOPS)", fontsize=fontSize * 2)
ax1.set_ylim(0, 420)
ax1.set_xticks([0, 1.8])
ax1.set_xticklabels(methods1, fontsize=fontSize * 1.7)
methods2 = ["FlashAttention2", "Ours(Mixed Precision)"]
speeds2 = [88, 73]
x_pos2 = np.arange(len(methods2))
bar_width2 = 0.8
bars21 = ax2.bar(0, speeds2[0], bar_width2, color="tomato", label="FlashAttention2")
bars22 = ax2.bar(
    1 + bar_width2,
    speeds2[1],
    bar_width2,
    color="mediumaquamarine",
    label="Ours(Mixed Precision)",
)
ax2.text(
    x_pos2[0],
    speeds2[0] + 0.5,
    str(speeds2[0]) + "s",
    ha="center",
    va="bottom",
    fontsize=fontSize * 1.4,
    color="black",
)
ax2.text(
    x_pos2[1] + bar_width2,
    speeds2[1] + 0.5,
    str(speeds2[1]) + "s",
    ha="center",
    va="bottom",
    fontsize=fontSize * 1.4,
    color="black",
)
ax2.annotate(
    "1.2x",
    xytext=(0.2, 94),
    xy=(1.3, 69),
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=fontSize * 1.5,
    color="red",
)
ax2.legend(loc="upper left", fontsize=fontSize * 1.8)
ax2.set_ylabel("End to End Speed (s)", fontsize=fontSize * 2)
ax2.set_ylim(0, 160)
ax1.yaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_xticks([0, 1.8])
ax2.set_xticklabels(methods2, fontsize=fontSize * 1.7)
plt.tight_layout()
plt.savefig("kernel_speed_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("kernel_speed_comparison.svg", dpi=300, bbox_inches="tight")
plt.savefig("kernel_speed_comparison.pdf", dpi=300)
plt.show()
