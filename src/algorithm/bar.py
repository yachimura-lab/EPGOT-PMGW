import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# ----------------------------
# パラメータ
# ----------------------------
Lambda_list = np.arange(0.125, 0.5625, 0.0625)
reg_list = np.arange(0.01, 0.31, 0.1)

print("C1=", C1)
print("C2=", C2)

k = 2 * max(np.max(C1), np.max(C2))
cost1 = C1 / k
cost2 = C2 / k
M = gaussian_W(Gmm_1, Gmm_2)
M = M / np.max(M)

# ----------------------------
# Figure 作成
# ----------------------------
fig, axes = plt.subplots(len(reg_list), len(Lambda_list), figsize=(30, 30))

# ----------------------------
# カラーマップ設定
# ----------------------------
cmap = cm.get_cmap("viridis")
norm = Normalize(vmin=0, vmax=np.max(M))  # R[i,j] の最大値に応じてスケーリング

# ----------------------------
# メインループ
# ----------------------------
for row, reg in enumerate(reg_list):
    for col, Lambda in enumerate(Lambda_list):
        ax = axes[row, col]
        plt.sca(ax)

        # --- GMM 描画 ---
        display_gmm(
            [Gmm_1.K, Gmm_1.weights, Gmm_1.comp_mean, Gmm_1.comp_cov],
            n=100,
            ax=-12,
            bx=12,
            ay=-12,
            by=12,
            cmap="Reds",
        )
        display_gmm(
            [Gmm_2.K, Gmm_2.weights, Gmm_2.comp_mean, Gmm_2.comp_cov],
            n=100,
            ax=-12,
            bx=12,
            ay=-12,
            by=12,
            cmap="Blues",
        )

        # --- Entropic Partial OT ---
        R = entropic_partial_ot(
            a=Gmm_1.weights,
            b=Gmm_2.weights,
            M=M,
            Lambda=Lambda,
            reg=reg,
            numItermax=1000,
            stopThr=1e-9,
            remove_dummy=True,
        )
        mass_partial = np.sum(R)
        print(f"Lambda={Lambda}, reg={reg}, mass_partial={mass_partial:.4f}")

        # --- Coupling lines with color by weight ---
        eps = 2.0 * 1e-3
        for i in range(Gmm_1.K):
            for j in range(Gmm_2.K):
                w = R[i, j]
                if w < eps:
                    continue

                p1 = Gmm_1.comp_mean[i]
                p2 = Gmm_2.comp_mean[j]

                lw = 0.1 + 7 * w
                color = cmap(norm(w))

                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], c=color, linewidth=lw, alpha=0.7
                )

        # --- Centers ---
        for i, (x, y) in enumerate(Gmm_1.comp_mean):
            ax.scatter(x, y, c="black", s=60)
            ax.text(x + 0.05, y + 0.05, f"1-{i}", color="black", fontsize=12)

        for j, (x, y) in enumerate(Gmm_2.comp_mean):
            ax.scatter(x, y, c="black", s=60)
            ax.text(x + 0.05, y + 0.05, f"2-{j}", color="black", fontsize=12)

# ----------------------------
# カラーバー配置（下段）
# ----------------------------
fig.subplots_adjust(
    bottom=0.10,
    top=0.95,
    left=0.05,
    right=0.95,
    wspace=0.15,  # 列間を少し狭く
    hspace=0.05,  # 行間を最小化
)

# 下段に横カラーバー
cbar_ax = fig.add_axes([0.05, 0.03, 0.90, 0.02])
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$w_{ij}$ (coupling weight)", fontsize=12)

# tight_layout で調整（カラーバーと干渉しないように下端を確保）
plt.tight_layout(rect=[0, 0.05, 1, 1])

# 保存＆表示
plt.savefig("fig_lambda_reg_sweep_optimized.eps", format="eps")
plt.show()
