import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cpp_engine

from matplotlib.colors import LogNorm
import scipy.stats as sps


# --- 1. Probability density evaluation ---
def densite_theorique2d(mu, Sigma, alpha, x):
    alpha = np.atleast_1d(alpha)
    K = mu.shape[0]
    y = np.zeros(x.shape[0])
    for j in range(K):
        # Make sure the covariance matrix and the mean have the right shape
        s = np.array(Sigma[j]).reshape(2, 2) + np.eye(2) * 1e-6
        m = np.array(mu[j]).flatten()
        try:
            y += alpha[j] * sps.multivariate_normal.pdf(x, mean=m, cov=s)
        except Exception:
            continue
    return y


# --- 2. Plotting helper (with a widened distribution tail) ---
def display_gmm(
    gmm,
    n=100,
    ax=-12,
    bx=12,
    ay=-12,
    by=12,
    cmap="magma",
    axis=None,
    alpha=0.35,
    draw_scatter=True,
    num_levels=4,
):
    if axis is None:
        axis = plt.gca()
    pi, mu, S = np.array(gmm[0]), np.array(gmm[1]), np.array(gmm[2])

    x = np.linspace(ax, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    XX = np.column_stack([X.ravel(), Y.ravel()])

    Z = densite_theorique2d(mu, S, pi, XX).reshape(X.shape)
    Z = np.maximum(Z, 1e-20)
    Zmax = Z.max()

    # --- Tuning note: widen the tails by lowering lower_bound ---
    # It used to be 1e-2; 1e-4 makes the faint edges of the distribution visible.
    lower_bound = 1e-4
    levels = np.logspace(np.log10(Zmax * lower_bound), np.log10(Zmax), num_levels)
    norm = LogNorm(vmin=Zmax * lower_bound, vmax=Zmax)

    axis.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, alpha=alpha)
    axis.contour(
        X, Y, Z, levels=levels, colors="k", linewidths=0.5, alpha=min(alpha * 1.5, 1.0)
    )

    if draw_scatter:
        s_alpha = min(alpha * 2, 1.0)
        axis.scatter(
            mu[:, 0], mu[:, 1], c="white", s=8, edgecolors="k", zorder=10, alpha=s_alpha
        )

    axis.set_aspect("equal")
    axis.set_xlim(ax, bx)
    axis.set_ylim(ay, by)


# --- 3. Initial data ---
means_A = np.array([[-9.0, -9.0], [2.0, -9.0]])
covs_A = 0.2 * np.array([[[1, 0.2], [0.2, 1]], [[1, -0.1], [-0.1, 1]]])
alpha_A = np.array([0.5, 0.5])

means_B = np.array([[-6.5, 2.5], [3.5, 2.5], [9.0, 7.5]])
covs_B = 0.2 * np.array(
    [[[1, -0.1], [-0.1, 1]], [[1, 0.2], [0.2, 1]], [[1, 0.0], [0.0, 1]]]
)
alpha_B = np.array([0.45, 0.45, 0.1])

gmm_source = [alpha_A, means_A, covs_A]
gmm_target = [alpha_B, means_B, covs_B]

# Parameter grid
lambdas = [0.1, 0.25, 0.4, 0.5]
regs = [0.01, 1, 10]

# Pre-compute the cost matrix
M = cpp_engine.compute_cost(means_A, list(covs_A), means_B, list(covs_B))
M_normalized = M / np.max(M)

# Make sure the output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

grid_results = []

# --- 4. Generate the animation matrix ---
for lb_val in lambdas:
    row_files = []
    for rg_val in regs:
        P = cpp_engine.solve_partial_ot(
            alpha_A, alpha_B, M_normalized, lb_val, rg_val, 1000, 1e-9
        )
        P_norm = P / (np.sum(P) + 1e-10)

        fig, ax = plt.subplots(figsize=(5, 5))
        n_steps = 30
        ts = np.linspace(0, 1, n_steps)

        def update(frame, current_lb=lb_val, current_rg=rg_val, current_P=P_norm):
            ax.clear()
            t = ts[frame]

            # Background reference contours
            display_gmm(
                gmm_source,
                axis=ax,
                n=60,
                cmap="Reds",
                alpha=0.08,
                draw_scatter=False,
                num_levels=3,
            )
            display_gmm(
                gmm_target,
                axis=ax,
                n=60,
                cmap="Blues",
                alpha=0.08,
                draw_scatter=False,
                num_levels=3,
            )

            if t < 1e-9:
                current_gmm, current_cmap, current_alpha = gmm_source, "Reds", 0.5
            elif t > 1 - 1e-9:
                current_gmm, current_cmap, current_alpha = gmm_target, "Blues", 0.5
            else:
                interp_means, interp_covs, interp_weights = [], [], []
                for r in range(len(alpha_A)):
                    for c in range(len(alpha_B)):
                        if current_P[r, c] > 1e-3:
                            res = cpp_engine.interpolate(
                                means_A[r], means_B[c], covs_A[r], covs_B[c], t
                            )
                            interp_means.append(res["mean"])
                            interp_covs.append(res["cov"])
                            interp_weights.append(current_P[r, c])
                current_gmm = [
                    np.array(interp_weights),
                    np.array(interp_means),
                    np.array(interp_covs),
                ]
                current_cmap, current_alpha = "magma", 0.5

            display_gmm(
                current_gmm,
                axis=ax,
                n=90,
                cmap=current_cmap,
                alpha=current_alpha,
                num_levels=4,
            )
            ax.set_title(f"L={current_lb}, R={current_rg} | t={t:.2f}", fontsize=10)

        # --- Tuning note: interval=200 slows the playback down ---
        ani = FuncAnimation(fig, update, frames=len(ts), interval=200)

        filename = f"matrix_L{str(lb_val).replace('.', '_')}_R{str(rg_val).replace('.', '_')}.gif"
        save_path = os.path.join(output_dir, filename)

        ani.save(save_path, writer="pillow")
        plt.close(fig)
        row_files.append(save_path)
        print(f"Saved: {save_path}")

    grid_results.append(row_files)

# --- 5. Generate the HTML report ---
html_path = "compare_report.html"
with open(html_path, "w") as f:
    f.write(
        r"<html><head><style>table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:10px;text-align:center;font-family:sans-serif;}.label{font-weight:bold;background:#f4f4f4;}img{max-width:300px;}</style></head><body>"
    )
    f.write("<h1>Partial OT Interpolation Matrix</h1>")
    f.write("<table>")
    f.write(r"<tr><td class='label'>Lambda \ reg</td>")
    for rg in regs:
        f.write(f"<td class='label'>reg = {rg}</td>")
    f.write("</tr>")
    for i, lb in enumerate(lambdas):
        f.write(f"<tr><td class='label'>Lambda = {lb}</td>")
        for gif_path in grid_results[i]:
            # Reference the images by relative path
            f.write(
                f"<td><img src='{gif_path}'><br><small>{os.path.basename(gif_path)}</small></td>"
            )
        f.write("</tr>")
    f.write("</table></body></html>")

print(f"\nMatrix Complete: '{html_path}'")
