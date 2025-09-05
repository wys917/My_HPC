import matplotlib.pyplot as plt


N = [500, 1000, 1500, 2000]
Gflops = [0.24072, 0.85795, 1.6817, 2.5377]


plt.figure(figsize=(6,4))
plt.plot(N, Gflops, marker='o', linestyle='-', linewidth=2)


plt.title("N vs. Gflops (HPL Benchmark)", fontsize=14)
plt.xlabel("Problem Size N", fontsize=12)
plt.ylabel("Performance (Gflops)", fontsize=12)


for x, y in zip(N, Gflops):
    plt.text(x, y + 0.05, f"{y:.2f}", ha='center', fontsize=10)


plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# 保存图片到assets目录
plt.savefig("assets/hpl_n_vs_gflops_plot.png", dpi=300, bbox_inches='tight')
plt.show()
