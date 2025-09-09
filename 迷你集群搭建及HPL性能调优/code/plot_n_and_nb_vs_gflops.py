import matplotlib.pyplot as plt


N = [500, 1000, 1500, 2000]
Gflops_N = [0.24072, 0.85795, 1.6817, 2.5377]


NB = [32, 64, 96, 128, 160, 192, 224, 256]
Gflops_NB = [2.0376, 2.2678, 2.6805, 2.5053, 2.6498, 2.8955, 3.1339, 3.0896]


fig, axes = plt.subplots(1, 2, figsize=(12, 4))


axes[0].plot(N, Gflops_N, marker='o', linestyle='-', linewidth=2)
axes[0].set_title("N vs. Gflops", fontsize=14)
axes[0].set_xlabel("Problem Size N", fontsize=12)
axes[0].set_ylabel("Performance (Gflops)", fontsize=12)
for x, y in zip(N, Gflops_N):
    axes[0].text(x, y + 0.05, f"{y:.2f}", ha='center', fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.6)


axes[1].plot(NB, Gflops_NB, marker='o', linestyle='-', linewidth=2, color='orange')
axes[1].set_title("NB vs. Gflops (N=2000, P=2, Q=5)", fontsize=14)
axes[1].set_xlabel("Block Size NB", fontsize=12)
for x, y in zip(NB, Gflops_NB):
    axes[1].text(x, y + 0.05, f"{y:.2f}", ha='center', fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("assets/compare.png", dpi=300, bbox_inches='tight')
plt.show()
