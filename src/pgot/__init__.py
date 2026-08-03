"""PGOT public API.

Implementations live in responsibility-specific modules; imports from the
package root remain available for compatibility with existing notebooks.
"""

from .gaussian import (
    Gaussian,
    GaussianMixture,
    GaussianW2,
    I,
    T_map_Gaussian,
    T_map_Gaussian_rand,
    T_mean,
    T_rand,
    gaussian_W,
    gaussian_cost_matrix,
    gaussian_w2_squared,
    gmm_transform,
    grad,
    grad_gauss,
    proj_gradient_descent,
    proj_stiefel,
    sample_from_gmm,
)
from .mew import EW2, MEW2, MEW2_GM, MEW2_coup, aEW2
from .mgw import MGW2, MGW2_GM, MGW2_GM_coup, MGW2_coup, aMGW2_GM, aMGW2_GM_coup
from .partial_mgw import (
    alignment_grad,
    alignment_loss,
    matched_statistics,
    pMGW2_coup,
    partial_mgw_barycentric_map,
    partial_projected_gradient_descent,
    validate_partial_coupling,
)
from .partial_ot import (
    compute_T_X_to_Z,
    compute_T_X_to_Z_C,
    compute_monge_map_matrix,
    densite_theorique2d,
    display_gmm,
    entropic_partial_ot,
    normalize_cost,
    partial_wasserstein_lagrange_entropic,
    to_numpy,
)
from .reproducibility import (
    DEFAULT_SEED,
    DEFAULT_SOURCE_DATE_EPOCH,
    freeze_output_timestamps,
    get_rng,
    set_reproducible,
    set_seed,
)

__all__ = [
    # Reproducibility
    "DEFAULT_SEED",
    "DEFAULT_SOURCE_DATE_EPOCH",
    "freeze_output_timestamps",
    "get_rng",
    "set_reproducible",
    "set_seed",
    # Gaussian models, geometry, and maps
    "Gaussian",
    "GaussianMixture",
    "GaussianW2",
    "I",
    "T_map_Gaussian",
    "T_map_Gaussian_rand",
    "T_mean",
    "T_rand",
    "gaussian_W",
    "gaussian_cost_matrix",
    "gaussian_w2_squared",
    "gmm_transform",
    "grad",
    "grad_gauss",
    "proj_gradient_descent",
    "proj_stiefel",
    "sample_from_gmm",
    # Mixture GW
    "MGW2",
    "MGW2_GM",
    "MGW2_GM_coup",
    "MGW2_coup",
    "aMGW2_GM",
    "aMGW2_GM_coup",
    # Embedded W2
    "EW2",
    "MEW2",
    "MEW2_GM",
    "MEW2_coup",
    "aEW2",
    # Partial OT and Figure 5 helpers
    "compute_T_X_to_Z",
    "compute_T_X_to_Z_C",
    "compute_monge_map_matrix",
    "densite_theorique2d",
    "display_gmm",
    "entropic_partial_ot",
    "normalize_cost",
    "partial_wasserstein_lagrange_entropic",
    "to_numpy",
    # Partial mixture GW
    "alignment_grad",
    "alignment_loss",
    "matched_statistics",
    "pMGW2_coup",
    "partial_mgw_barycentric_map",
    "partial_projected_gradient_descent",
    "validate_partial_coupling",
]
