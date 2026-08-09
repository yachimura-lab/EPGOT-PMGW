"""Section 7 constants, duplicated here so the tests are self-contained.

The notebooks are the authority for these values (issue #21). This module is
a deliberate second copy used only by the tests;
``test_notebook_specification.py`` asserts that the two agree, so drift in
either place fails the suite rather than passing silently.
"""

import numpy as np

from pgot import GaussianMixture

# Section 7.1
SOURCE_WEIGHTS = np.array([0.5, 0.5])
SOURCE_MEANS = np.array([[-8.0, -5.0], [1.0, -8.0]])
SOURCE_COVARIANCES = 0.2 * np.array(
    [[[1.0, 0.2], [0.2, 1.0]], [[1.0, -0.1], [-0.1, 1.0]]]
)
TARGET_WEIGHTS = np.array([0.45, 0.45, 0.1])
TARGET_MEANS = np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]])
TARGET_COVARIANCES = 0.2 * np.array(
    [
        [[1.0, -0.1], [-0.1, 1.0]],
        [[1.0, 0.2], [0.2, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]
)

# Section 7.1.1--7.1.5 sweeps
FIGURE1_LAMBDA, FIGURE1_EPSILON = 0.3, 0.01
FIGURE2_LAMBDAS, FIGURE2_EPSILON = [0.0, 0.125, 0.25, 0.375, 0.5], 0.01
FIGURE3_LAMBDAS = [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5]
FIGURE3_EPSILONS = [0.01, 0.11, 0.21]
FIGURE4_CONFIGURATIONS = [[0.25, 0.03], [0.25, 0.1], [0.5, 0.03]]
FIGURE4_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
FIGURE5_LAMBDAS, FIGURE5_EPSILON = [0.25, 0.5], 0.03
FIGURE5_TIMES = [0.2, 0.4, 0.6, 0.8, 1.0]
SAMPLES_PER_MIXTURE = 2000

# Section 7.2
COMPONENTS, POINTS, SIGMA = 6, 300, 0.2
RADIUS, HEIGHT = 4.0, 10.0
NOISE_POINTS, NOISE_RADIUS = 10, 0.3
SOURCE_COMPONENTS, TARGET_COMPONENTS = 6, 7
PAPER_LAMBDA = 0.01


def paper_mixtures():
    """The Section 7.1 source and target mixtures."""
    return (
        GaussianMixture(SOURCE_WEIGHTS, SOURCE_MEANS, SOURCE_COVARIANCES),
        GaussianMixture(TARGET_WEIGHTS, TARGET_MEANS, TARGET_COVARIANCES),
    )


def normalized_cross_cost(source, target):
    """Normalized component cost (7.1) and the scale it was divided by."""
    from pgot import gaussian_W

    raw = gaussian_W(source, target)
    scale = float(raw.max())
    return raw / scale, scale


def point_cloud_data(seed=None):
    """The Section 7.2 rings and noise, taken from the notebook itself.

    The notebook is the authority for the geometry, so the test executes its
    data cell instead of keeping a second implementation that could drift.
    """
    import io
    import json
    from contextlib import redirect_stdout
    from pathlib import Path

    notebook = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "notebook"
            / "PMGW_corrected.ipynb"
        ).read_text()
    )
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "figure67_data = {" in "".join(c["source"])
    )
    namespace = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(cell, "PMGW_corrected:data", "exec"), namespace)
    data = namespace["figure67_data"]
    if seed is not None and data["seed"] != seed:
        raise AssertionError(
            f"notebook seeds the data with {data['seed']}, not {seed}"
        )
    return data
