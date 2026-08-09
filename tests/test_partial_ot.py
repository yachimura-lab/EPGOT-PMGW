import unittest

import numpy as np

from pgot import (
    compute_T_X_to_Z,
    compute_T_X_to_Z_C,
    entropic_partial_coupling,
    entropic_partial_displacement_interpolation,
    paper_gaussian_mixtures,
    sample_from_gmm,
    submixture_pdf,
)


class PartialOTRegressionTests(unittest.TestCase):
    def setUp(self):
        self.source, self.target = paper_gaussian_mixtures()

    def test_figure5_epot_mass_and_residual(self):
        expected = {0.25: 0.8875742, 0.5: 0.9937141}
        for paper_lambda, expected_mass in expected.items():
            coupling, info = entropic_partial_coupling(
                self.source,
                self.target,
                epsilon=0.03,
                Lambda=paper_lambda,
                log=True,
            )
            self.assertAlmostEqual(coupling.sum(), expected_mass, places=6)
            self.assertTrue(info["converged"])
            self.assertLess(info["row_residual"], 1e-8)
            self.assertLess(info["column_residual"], 1e-8)

    def test_figure4_submixture_preserves_partial_mass(self):
        coupling = entropic_partial_coupling(
            self.source, self.target, epsilon=0.03, Lambda=0.25
        )
        (weights, means, covariances), info = (
            entropic_partial_displacement_interpolation(
                self.source,
                self.target,
                0.5,
                epsilon=0.03,
                Lambda=0.25,
                coupling=coupling,
                log=True,
            )
        )
        self.assertEqual(info["solver"], "reused coupling")
        self.assertAlmostEqual(weights.sum(), coupling.sum(), places=12)

        grid_axis = np.linspace(-15.0, 15.0, 250)
        gx, gy = np.meshgrid(grid_axis, grid_axis)
        density = submixture_pdf(
            np.column_stack([gx.ravel(), gy.ravel()]),
            weights,
            means,
            covariances,
        ).reshape(gx.shape)
        integral = np.trapezoid(np.trapezoid(density, grid_axis, axis=1), grid_axis)
        self.assertAlmostEqual(integral, coupling.sum(), places=3)

    def test_legacy_map_now_uses_matched_density(self):
        rng_source = np.random.default_rng(10)
        rng_target = np.random.default_rng(11)
        source_points = sample_from_gmm(
            self.source.weights,
            self.source.comp_mean,
            self.source.comp_cov,
            120,
            rng=rng_source,
        )
        target_points = sample_from_gmm(
            self.target.weights,
            self.target.comp_mean,
            self.target.comp_cov,
            120,
            rng=rng_target,
        )
        legacy = compute_T_X_to_Z(
            source_points,
            target_points,
            2,
            3,
            epsilon=0.03,
            Lambda=0.25,
            random_state=42,
        )
        canonical = compute_T_X_to_Z_C(
            source_points,
            target_points,
            2,
            3,
            epsilon=0.03,
            Lambda=0.25,
            random_state=42,
        )
        np.testing.assert_allclose(legacy, canonical, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
