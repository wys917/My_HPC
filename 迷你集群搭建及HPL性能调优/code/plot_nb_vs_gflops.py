import matplotlib.pyplot as plt


NB = [32, 64, 96, 128, 160, 192, 224, 256]
Gflops = [2.0376, 2.2678, 2.6805, 2.5053, 2.6498, 2.8955, 3.1339, 3.0896]


plt.figure(figsize=(6,4))
plt.plot(NB, Gflops, marker='o', linestyle='-', linewidth=2)


plt.title("NB vs. Gflops (HPL Benchmark, N=2000, P=2, Q=5)", fontsize=14)
plt.xlabel("Block Size NB", fontsize=12)
plt.ylabel("Performance (Gflops)", fontsize=12)


for x, y in zip(NB, Gflops):
    plt.text(x, y + 0.05, f"{y:.2f}", ha='center', fontsize=10)


plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("assets/plot_nb_vs_gflops.png", dpi=300, bbox_inches='tight')
plt.show()
