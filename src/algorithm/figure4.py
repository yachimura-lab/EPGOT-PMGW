import sys

import os

import numpy as np

import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal

from matplotlib.colors import LogNorm


# 设置路径

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cpp_engine


class GaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = np.array(weights)

        self.comp_mean = np.array(means)

        self.comp_cov = np.array(covs)

        self.K = len(weights)

    def pdf(self, x):
        pdf_sum = np.zeros(x.shape[0])

        for k in range(self.K):
            pdf_sum += self.weights[k] * multivariate_normal.pdf(
                x, self.comp_mean[k], self.comp_cov[k]
            )

        return pdf_sum


def display_gmm(
    gmm, n=400, ax_lim=-12, bx_lim=12, ay_lim=-12, by_lim=12, cmap="Reds", axis=None
):
    if axis is None:
        fig, axis = plt.subplots()

    x = np.linspace(ax_lim, bx_lim, n)

    y = np.linspace(ay_lim, by_lim, n)

    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y)).reshape(-1, 2)

    Z = gmm.pdf(pos).reshape(n, n)

    # 关键：强制 1:1 比例，否则椭圆会看起来像被挤压的形状

    axis.set_aspect("equal", adjustable="box")

    z_max = max(Z.max(), 1e-8)

    levels = np.logspace(np.log10(z_max * 1e-3), np.log10(z_max), 7)

    axis.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
        alpha=0.5,
    )

    return axis


def entropic_partial_barycentric_interpolation(gmm_1, gmm_2, t, Lambda, reg):
    M = np.zeros((gmm_1.K, gmm_2.K))

    for i in range(gmm_1.K):
        for j in range(gmm_2.K):
            M[i, j] = cpp_engine.GaussianW2(
                gmm_1.comp_mean[i],
                gmm_2.comp_mean[j],
                gmm_1.comp_cov[i],
                gmm_2.comp_cov[j],
            )

    M /= np.max(M)

    W = cpp_engine.entropic_partial_ot(
        gmm_1.weights, gmm_2.weights, M, Lambda, reg, 1000, 1e-7
    )

    new_w, new_m, new_c = [], [], []

    alpha_vec = np.array([1.0 - t, float(t)], dtype=np.float64)

    for i in range(gmm_1.K):
        for j in range(gmm_2.K):
            if W[i, j] > 1e-5:
                # 调用 C++ 迭代重心

                m_ij, c_ij = cpp_engine.GaussianBarycenterW2(
                    [gmm_1.comp_mean[i], gmm_2.comp_mean[j]],
                    [gmm_1.comp_cov[i], gmm_2.comp_cov[j]],
                    alpha_vec,
                    20,
                )

                new_w.append(W[i, j])

                new_m.append(m_ij)

                new_c.append(c_ij)

    return GaussianMixture(np.array(new_w) / np.sum(new_w), new_m, new_c)


def main():
    alpha_A = np.array([0.5, 0.5])
    means_A = np.array([[-8.0, -5.0], [1.0, -8.0]])
    covs_A = 0.2 * np.array([[[1, 0.2], [0.2, 1]], [[1, -0.1], [-0.1, 1]]])
    Gmm_1 = GaussianMixture(alpha_A, means_A, covs_A)

    alpha_B = np.array([0.45, 0.45, 0.1])
    means_B = np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]])
    covs_B = 0.2 * np.array(
        [[[1, -0.1], [-0.1, 1]], [[1, 0.2], [0.2, 1]], [[1, 0.0], [0.0, 1]]]
    )
    Gmm_2 = GaussianMixture(alpha_B, means_B, covs_B)
    # --- 設定 ---
    t_steps = 5
    t_list = np.linspace(0, 1, t_steps)
    Lambda = 0.25
    reg = 0.01

    # 1行5列のキャンバスを作成
    fig, axes = plt.subplots(1, t_steps, figsize=(20, 4))

    if t_steps == 1:
        axes = [axes]

    print("Calculating and rendering steps...")
    for i, t in enumerate(t_list):
        ax = axes[i]

        # 背景として始点(Red)と終点(Blue)を薄く表示
        display_gmm(Gmm_1, cmap="Reds", axis=ax)
        display_gmm(Gmm_2, cmap="Blues", axis=ax)

        # Partial OT による補間計算
        gmm_t = entropic_partial_barycentric_interpolation(Gmm_1, Gmm_2, t, Lambda, reg)

        # 補間結果を表示 (Viridis)
        display_gmm(gmm_t, cmap="viridis", axis=ax)

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_title(f"t = {t:.2f}")

        # 見栄えのため軸ラベルを消去（任意）
        if i > 0:
            ax.set_yticklabels([])

    plt.tight_layout()

    # 保存
    output_path = "output/interpolation_line_L025_R001.eps"
    os.makedirs("output", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Success: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
