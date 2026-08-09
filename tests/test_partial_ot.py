import unittest

import numpy as np

from paper_setup import paper_mixtures
from pgot import (
    compute_T_X_to_Z,
    compute_T_X_to_Z_C,
    entropic_partial_coupling,
    entropic_partial_displacement_interpolation,
    sample_from_gmm,
    submixture_pdf,
)


class PartialOTRegressionTests(unittest.TestCase):
    def setUp(self):
        self.source, self.target = paper_mixtures()

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

    def test_no_penalty_stands_in_for_the_balanced_problem(self):
        # A finite Lambda never makes the dummy node empty at eps > 0, so a
        # default penalty would hand back a partial coupling while the caller
        # believed they had asked for balanced entropic OT.
        with self.assertRaises(TypeError):
            entropic_partial_coupling(self.source, self.target, epsilon=0.03)
        with self.assertRaisesRegex(ValueError, "Lambda must be specified"):
            entropic_partial_coupling(
                self.source, self.target, epsilon=0.03, Lambda=None
            )

    def test_the_superseded_solver_is_not_reachable_from_the_package_root(self):
        # It solves 'M - Lambda' with entropy on the real block only, so a
        # caller who reaches it from `from pgot import ...` gets a coupling
        # that is not the paper's (3.1) while every name around it is.
        import pgot
        import pgot.legacy

        self.assertNotIn("partial_wasserstein_lagrange_entropic", pgot.__all__)
        self.assertFalse(hasattr(pgot, "partial_wasserstein_lagrange_entropic"))
        # Still importable, so published results stay reproducible.
        with self.assertWarns(DeprecationWarning):
            pgot.legacy.partial_wasserstein_lagrange_entropic(
                np.array([0.5, 0.5]),
                np.array([0.5, 0.5]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                Lambda=0.5,
            )

    def test_both_figure5_entry_points_demand_the_penalty(self):
        # compute_T_X_to_Z and compute_T_X_to_Z_C are documented as
        # equivalent, so a default penalty on one and not the other would let
        # the same call give two different couplings.
        points = sample_from_gmm(
            self.source.weights,
            self.source.comp_mean,
            self.source.comp_cov,
            40,
            rng=np.random.default_rng(0),
        )
        for entry_point in (compute_T_X_to_Z, compute_T_X_to_Z_C):
            with self.subTest(entry_point=entry_point.__name__):
                with self.assertRaises(TypeError):
                    entry_point(points, points, 2, 2, epsilon=0.03)

    def test_the_dummy_node_keeps_mass_at_the_penalty_that_was_the_default(self):
        # The removed default was Lambda = max(M) = 1 on the normalized cost.
        # It leaves matched mass short of one, which is what made calling it
        # the balanced limit wrong.
        coupling = entropic_partial_coupling(
            self.source, self.target, epsilon=0.03, Lambda=1.0
        )
        self.assertLess(coupling.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
