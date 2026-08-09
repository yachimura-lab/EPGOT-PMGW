import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pgot import (
    FIGURE_PARAMETERS,
    MGW2_GM_coup,
    fit_figure67_problem,
    generate_figure67_data,
    gmm_transform,
    make_figure1,
    make_figure2,
    make_figure3,
    paper_gaussian_mixtures,
    solve_epot_panels,
    solve_figure67,
)


class CanonicalEPOTFigureTests(unittest.TestCase):
    def test_figure1_2_3_parameters_and_residuals(self):
        figure1 = solve_epot_panels([0.3], [0.01])
        figure2 = solve_epot_panels(
            FIGURE_PARAMETERS[2]["lambda"], [FIGURE_PARAMETERS[2]["epsilon"]]
        )
        difficult = solve_epot_panels([0.5], [0.01])

        self.assertEqual(len(figure1), 1)
        self.assertEqual([r["paper_lambda"] for r in figure2], [0, 0.125, 0.25, 0.375, 0.5])
        self.assertTrue(difficult[0]["fallback_used"])
        for record in figure1 + figure2 + difficult:
            self.assertTrue(record["converged"])
            self.assertLess(record["row_residual"], 1e-8)
            self.assertLess(record["column_residual"], 1e-8)

    def test_canonical_plot_layouts(self):
        records1 = solve_epot_panels([0.3], [0.01])
        records2 = solve_epot_panels(FIGURE_PARAMETERS[2]["lambda"], [0.01])
        records3 = solve_epot_panels(FIGURE_PARAMETERS[3]["lambda"], [0.11])
        figures = [
            make_figure1(records1),
            make_figure2(records2),
            make_figure3(records3 * 3),
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
        cls.data = generate_figure67_data()
        cls.problem = fit_figure67_problem(cls.data)
        cls.solved = solve_figure67(cls.problem)

    def test_geometry_is_independent_and_matches_paper_radius(self):
        source = self.data["source"]
        target_ring = self.data["target_ring"]
        self.assertAlmostEqual(np.hypot(*source[:, :2].T).mean(), 4.0, delta=0.05)
        self.assertAlmostEqual(
            np.hypot(*target_ring[:, :2].T).mean(), 4.0, delta=0.05
        )
        self.assertGreater(source[:, 2].std(), 0.1)
        self.assertFalse(np.allclose(source[:, :2], target_ring[:, :2]))
        self.assertEqual(len(self.data["noise"]), 10)

    def test_partial_methods_match_300_over_310(self):
        expected = 300 / 310
        self.assertAlmostEqual(self.solved["coupling_pgw"].sum(), expected, places=12)
        self.assertAlmostEqual(self.solved["coupling_pmgw"].sum(), expected, places=12)
        self.assertAlmostEqual(self.solved["lambda_point"], 0.01, places=12)
        self.assertAlmostEqual(self.solved["lambda_component"], 0.0132491708, places=9)

    def test_figure6_and_7_share_mixture_solve_ids(self):
        # Both figures are rendered from this one result object. The raw maps
        # and nearest-neighbour assignments therefore retain the same IDs.
        self.assertEqual(len(self.solved["solve_ids"]["MGW2"]), 64)
        self.assertEqual(len(self.solved["solve_ids"]["pMGW2"]), 64)
        self.assertEqual(len(self.solved["map_mgw"]), len(self.solved["index_mgw"]))
        self.assertEqual(len(self.solved["map_pmgw"]), len(self.solved["index_pmgw"]))

    def test_mgw_coupling_is_translation_invariant(self):
        source, target = paper_gaussian_mixtures()
        coupling = MGW2_GM_coup(source, target)
        shifted_source = gmm_transform(source, b=np.array([17.0, -4.0]))
        shifted_target = gmm_transform(target, b=np.array([-8.0, 11.0]))
        shifted = MGW2_GM_coup(shifted_source, shifted_target)
        np.testing.assert_allclose(coupling, shifted, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
