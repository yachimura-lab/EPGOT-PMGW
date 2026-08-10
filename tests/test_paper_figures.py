import unittest
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

import ot
import pgot.mgw as pgot_mgw
from lib import gromov
from pgot import paper_figures

from paper_setup import (
    FIGURE2_LAMBDAS,
    FIGURE3_EPSILONS,
    FIGURE3_LAMBDAS,
    PAPER_LAMBDA,
    SOURCE_COMPONENTS,
    TARGET_COMPONENTS,
    normalized_cross_cost,
    paper_mixtures,
    point_cloud_data,
)
from pgot import (
    DEFAULT_SEED,
    nearest_target_indices,
    GaussianMixture,
    MGW2_GM_coup,
    gaussian_cost_matrix,
    gmm_transform,
    gw_mean_distortion,
    make_figure1,
    make_figure2,
    make_figure3,
    solve_epot_panels,
    solve_point_cloud_matching,
)
from sklearn.mixture import GaussianMixture as SkGaussianMixture


SOURCE, TARGET = paper_mixtures()
COST, COST_SCALE = normalized_cross_cost(SOURCE, TARGET)


def panels(lambdas, epsilons):
    return solve_epot_panels(lambdas, epsilons, SOURCE, TARGET, COST, COST_SCALE)


def solved_point_clouds():
    data = point_cloud_data(DEFAULT_SEED)

    def fit(points, components):
        model = SkGaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=DEFAULT_SEED,
        ).fit(points)
        return GaussianMixture(model.weights_, model.means_, model.covariances_)

    source_gmm = fit(data["source"], SOURCE_COMPONENTS)
    target_gmm = fit(data["target"], TARGET_COMPONENTS)
    component_source = gaussian_cost_matrix(source_gmm)
    component_target = gaussian_cost_matrix(target_gmm)
    component_scale = float(max(component_source.max(), component_target.max()))
    point_source = ot.dist(data["source"], data["source"], metric="sqeuclidean")
    point_target = ot.dist(data["target"], data["target"], metric="sqeuclidean")
    point_scale = float(max(point_source.max(), point_target.max()))
    problem = {
        **data,
        "source_gmm": source_gmm,
        "target_gmm": target_gmm,
        "component_source_cost": component_source,
        "component_target_cost": component_target,
        "component_source_cost_normalized": component_source / component_scale,
        "component_target_cost_normalized": component_target / component_scale,
        "component_cost_scale": component_scale,
        "point_source_cost_normalized": point_source / point_scale,
        "point_target_cost_normalized": point_target / point_scale,
        "point_cost_scale": point_scale,
        "fit_random_state": DEFAULT_SEED,
    }
    p = ot.unif(len(data["source"]))
    q = ot.unif(len(data["target"]))
    coupling_gw = ot.gromov.gromov_wasserstein(
        problem["point_source_cost_normalized"],
        problem["point_target_cost_normalized"],
        p,
        q,
        loss_fun="square_loss",
    )
    coupling_mgw = MGW2_GM_coup(
        source_gmm,
        target_gmm,
        C1=problem["component_source_cost_normalized"],
        C2=problem["component_target_cost_normalized"],
    )
    lambda_point = PAPER_LAMBDA
    lambda_component = PAPER_LAMBDA
    solved = solve_point_cloud_matching(
        problem,
        coupling_gw=coupling_gw,
        coupling_mgw=coupling_mgw,
        lambda_point=lambda_point,
        lambda_component=lambda_component,
        method_parameters={
            "paper_lambda": PAPER_LAMBDA,
            "point_solver_lambda": lambda_point,
            "component_solver_lambda": lambda_component,
            "partial_init": "balanced coupling of the same representation",
        },
        runtime_gw=0.0,
        runtime_mgw_coupling=0.0,
    )
    return data, problem, solved


class CanonicalEPOTFigureTests(unittest.TestCase):
    def test_figure1_2_3_parameters_and_residuals(self):
        figure1 = panels([0.3], [0.01])
        figure2 = panels(FIGURE2_LAMBDAS, [0.01])
        difficult = panels([0.5], [0.01])

        self.assertEqual(len(figure1), 1)
        self.assertEqual([r["paper_lambda"] for r in figure2], [0, 0.125, 0.25, 0.375, 0.5])
        self.assertTrue(difficult[0]["fallback_used"])
        for record in figure1 + figure2 + difficult:
            self.assertTrue(record["converged"])
            self.assertLess(record["row_residual"], 1e-8)
            self.assertLess(record["column_residual"], 1e-8)

    def test_canonical_plot_layouts(self):
        records1 = panels([0.3], [0.01])
        records2 = panels(FIGURE2_LAMBDAS, [0.01])
        records3 = panels(FIGURE3_LAMBDAS, FIGURE3_EPSILONS)
        figures = [
            make_figure1(records1, SOURCE, TARGET),
            make_figure2(records2, SOURCE, TARGET),
            make_figure3(
                records3,
                SOURCE,
                TARGET,
                (len(FIGURE3_EPSILONS), len(FIGURE3_LAMBDAS)),
            ),
        ]
        # The last axes is the horizontal colorbar in each canonical figure.
        self.assertEqual(len(figures[0].axes), 2)
        self.assertEqual(len(figures[1].axes), 6)
        self.assertEqual(len(figures[2].axes), 22)
        for figure in figures:
            figure.clear()


class Figure67RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.problem, cls.solved = solved_point_clouds()

    def test_geometry_is_independent_and_matches_paper_radius(self):
        source = self.data["source"]
        target_ring = self.data["target_ring"]
        self.assertAlmostEqual(np.hypot(*source[:, :2].T).mean(), 4.0, delta=0.05)
        self.assertAlmostEqual(
            np.hypot(*target_ring[:, :2].T).mean(), 4.0, delta=0.05
        )
        self.assertGreater(source[:, 2].std(), 0.1)
        self.assertFalse(np.allclose(source[:, :2], target_ring[:, :2]))
        self.assertEqual(len(self.data["noise"]), 50)

    def test_partial_methods_match_300_over_350(self):
        expected = 300 / 350
        self.assertAlmostEqual(self.solved["coupling_pgw"].sum(), expected, places=12)
        self.assertAlmostEqual(self.solved["coupling_pmgw"].sum(), expected, places=12)
        self.assertAlmostEqual(self.solved["lambda_point"], 0.01, places=12)
        self.assertAlmostEqual(self.solved["lambda_component"], 0.01, places=12)

    def test_component_solve_needs_the_balanced_start(self):
        # Why solve_point_cloud_matching hands the balanced coupling to the
        # partial solver: from the solver's own default start, the component
        # problem stalls at the empty coupling at the paper's lambda, even
        # though the coupling reached from the balanced start scores strictly
        # lower. Without the start, Figure 6 and Figure 7 lose their pMGW panel.
        C1 = self.problem["component_source_cost_normalized"]
        C2 = self.problem["component_target_cost_normalized"]
        stalled = gromov.partial_gromov_ver1(
            C1,
            C2,
            self.problem["source_gmm"].weights,
            self.problem["target_gmm"].weights,
            Lambda=PAPER_LAMBDA,
            numItermax_gw=100,
            tol=1e-12,
            verbose=False,
        )
        self.assertAlmostEqual(stalled.sum(), 0.0, places=12)

        def objective(coupling):
            mass = coupling.sum()
            if mass <= 0.0:
                return 0.0
            distortion = gw_mean_distortion(C1, C2, coupling) * mass**2
            return distortion - 2.0 * PAPER_LAMBDA * mass**2

        self.assertLess(objective(self.solved["coupling_pmgw"]), objective(stalled))

    def test_figure6_is_the_nearest_neighbour_view_of_figure7(self):
        # Figure 7 shows map_*, Figure 6 shows index_*. If a method were solved
        # a second time the two would no longer be tied by nearest-neighbour
        # assignment on the same target cloud.
        target = self.data["target"]
        for name in ["mgw", "pmgw", "gw", "pgw"]:
            with self.subTest(method=name):
                np.testing.assert_array_equal(
                    self.solved[f"index_{name}"],
                    nearest_target_indices(self.solved[f"map_{name}"], target),
                )

    def test_balanced_solvers_are_never_called_inside_the_helper(self):
        # The caller already solved GW and MGW to convert the paper's lambda.
        # Making both balanced solvers explode proves the helper reuses those
        # couplings instead of quietly solving them a second time.
        def explode(*args, **kwargs):
            raise AssertionError("a balanced solver ran inside the helper")

        _, problem, _ = solved_point_clouds()
        with mock.patch.object(paper_figures, "MGW2_GM_coup", explode), mock.patch.object(
            pgot_mgw, "MGW2_GM_coup", explode
        ), mock.patch.object(ot.gromov, "gromov_wasserstein", explode):
            solved = solve_point_cloud_matching(
                problem,
                coupling_gw=self.solved["coupling_gw"],
                coupling_mgw=self.solved["coupling_mgw"],
                lambda_point=self.solved["lambda_point"],
                lambda_component=self.solved["lambda_component"],
                method_parameters={},
                runtime_gw=1.25,
                runtime_mgw_coupling=2.5,
            )
        np.testing.assert_array_equal(
            solved["coupling_gw"], self.solved["coupling_gw"]
        )
        np.testing.assert_array_equal(
            solved["coupling_mgw"], self.solved["coupling_mgw"]
        )
        self.assertEqual(solved["runtime_seconds"]["GW2"], 1.25)
        self.assertEqual(solved["runtime_seconds"]["MGW2_coupling"], 2.5)

    def test_mgw_coupling_is_translation_invariant(self):
        source, target = SOURCE, TARGET
        coupling = MGW2_GM_coup(source, target)
        shifted_source = gmm_transform(source, b=np.array([17.0, -4.0]))
        shifted_target = gmm_transform(target, b=np.array([-8.0, 11.0]))
        shifted = MGW2_GM_coup(shifted_source, shifted_target)
        np.testing.assert_allclose(coupling, shifted, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
