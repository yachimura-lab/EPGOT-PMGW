"""The notebooks must state Section 7 in full, and agree with each other.

Issue #21 moved every value the paper asserts out of ``pgot`` and into the
notebook cells, so that the published page can be checked against the paper.
That leaves the Section 7.1 mixtures duplicated across three notebooks and
copied once more into ``paper_setup`` for the tests. These checks are what
keeps the copies honest.
"""

import ast
import json
import re
import unittest
from pathlib import Path

import numpy as np

import paper_setup as spec

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "notebook"
LIBRARY_DIR = ROOT / "src" / "pgot"
MIXTURE_NOTEBOOKS = ["figure1_2_3", "figure4", "figure5"]


def source_of(name):
    notebook = json.loads((NOTEBOOK_DIR / f"{name}.ipynb").read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def literal(text, name):
    """Value of a top-level ``name = <literal>`` assignment in ``text``."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned as a literal")


def mixture_arguments(text, variable):
    """The three arrays passed to ``variable = GaussianMixture(...)``."""
    for node in ast.parse(text).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "GaussianMixture"
        ):
            return [np.asarray(eval_array(a)) for a in node.value.args]
    raise AssertionError(f"{variable} is not built from a GaussianMixture literal")


def eval_array(node):
    """Evaluate ``np.array([...])`` and ``scalar * np.array([...])``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return ast.literal_eval(node.left) * np.asarray(eval_array(node.right))
    if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "array":
        return np.asarray(ast.literal_eval(node.args[0]))
    raise AssertionError(f"unsupported array expression: {ast.dump(node)[:80]}")


class NotebooksStateSection71(unittest.TestCase):
    def test_every_notebook_uses_the_same_mixtures(self):
        for name in MIXTURE_NOTEBOOKS:
            text = source_of(name)
            with self.subTest(notebook=name):
                weights, means, covariances = mixture_arguments(text, "source")
                np.testing.assert_allclose(weights, spec.SOURCE_WEIGHTS)
                np.testing.assert_allclose(means, spec.SOURCE_MEANS)
                np.testing.assert_allclose(covariances, spec.SOURCE_COVARIANCES)
                weights, means, covariances = mixture_arguments(text, "target")
                np.testing.assert_allclose(weights, spec.TARGET_WEIGHTS)
                np.testing.assert_allclose(means, spec.TARGET_MEANS)
                np.testing.assert_allclose(covariances, spec.TARGET_COVARIANCES)


class NotebooksStateTheSweeps(unittest.TestCase):
    def test_figure1_2_3_sweeps(self):
        text = source_of("figure1_2_3")
        self.assertEqual(literal(text, "FIGURE1_LAMBDA"), spec.FIGURE1_LAMBDA)
        self.assertEqual(literal(text, "FIGURE1_EPSILON"), spec.FIGURE1_EPSILON)
        self.assertEqual(literal(text, "FIGURE2_LAMBDAS"), spec.FIGURE2_LAMBDAS)
        self.assertEqual(literal(text, "FIGURE2_EPSILON"), spec.FIGURE2_EPSILON)
        self.assertEqual(literal(text, "FIGURE3_LAMBDAS"), spec.FIGURE3_LAMBDAS)
        self.assertEqual(literal(text, "FIGURE3_EPSILONS"), spec.FIGURE3_EPSILONS)

    def test_figure4_configurations(self):
        text = source_of("figure4")
        self.assertEqual(literal(text, "configurations"), spec.FIGURE4_CONFIGURATIONS)
        self.assertEqual(literal(text, "times"), spec.FIGURE4_TIMES)

    def test_figure5_sweep_and_sample_count(self):
        text = source_of("figure5")
        self.assertEqual(literal(text, "FIGURE5_LAMBDAS"), spec.FIGURE5_LAMBDAS)
        self.assertEqual(literal(text, "FIGURE5_EPSILON"), spec.FIGURE5_EPSILON)
        self.assertEqual(literal(text, "FIGURE5_TIMES"), spec.FIGURE5_TIMES)
        self.assertEqual(
            literal(text, "SAMPLES_PER_MIXTURE"), spec.SAMPLES_PER_MIXTURE
        )

    def test_figure67_geometry_and_penalty(self):
        text = source_of("PMGW_corrected")
        for name, expected in [
            ("COMPONENTS", spec.COMPONENTS),
            ("POINTS", spec.POINTS),
            ("SIGMA", spec.SIGMA),
            ("RADIUS", spec.RADIUS),
            ("HEIGHT", spec.HEIGHT),
            ("NOISE_POINTS", spec.NOISE_POINTS),
            ("NOISE_RADIUS", spec.NOISE_RADIUS),
            ("SOURCE_COMPONENTS", spec.SOURCE_COMPONENTS),
            ("TARGET_COMPONENTS", spec.TARGET_COMPONENTS),
            ("PAPER_LAMBDA", spec.PAPER_LAMBDA),
        ]:
            with self.subTest(constant=name):
                self.assertEqual(literal(text, name), expected)


class LibraryHoldsNoPaperValues(unittest.TestCase):
    def test_no_section_seven_constants_in_pgot(self):
        # Values that only Section 7 would justify. A hit means a specification
        # value leaked back into the library.
        forbidden = re.compile(r"(?<![\w.])(0\.45|0\.1875|0\.3125|0\.4375|-8\.0|9\.0|7\.5)(?![\w.])")
        offenders = []
        for path in sorted(LIBRARY_DIR.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if forbidden.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()


class GeneratorMatchesCommittedNotebooks(unittest.TestCase):
    def test_build_notebooks_reproduces_every_cell(self):
        # The notebooks are build artifacts of tools/build_notebooks.py.
        # Editing one by hand, or editing the generator without re-running it,
        # must fail here rather than surface as a stale published page.
        import importlib.util

        spec_module = importlib.util.spec_from_file_location(
            "build_notebooks", ROOT / "tools" / "build_notebooks.py"
        )
        builder = importlib.util.module_from_spec(spec_module)
        spec_module.loader.exec_module(builder)

        for name, cells in builder.CELLS.items():
            expected = builder.build(cells)
            committed = json.loads((NOTEBOOK_DIR / name).read_text())
            with self.subTest(notebook=name):
                self.assertEqual(
                    [(c["cell_type"], "".join(c["source"])) for c in expected.cells],
                    [
                        (c["cell_type"], "".join(c["source"]))
                        for c in committed["cells"]
                    ],
                )


class SidecarsRecordEverySolve(unittest.TestCase):
    def test_all_panel_runtimes_are_positive(self):
        # A helper that stops timing a solve silently records 0.0 here, which
        # would make the provenance claim untrue while every figure still
        # looks correct.
        for name in ["figure1", "figure2", "figure3", "figure4", "figure5",
                     "figure6", "figure7"]:
            sidecar = json.loads(
                (NOTEBOOK_DIR / "metadata" / f"{name}.json").read_text()
            )
            for index, panel in enumerate(sidecar.get("panels", [])):
                if "runtime_seconds" not in panel:
                    continue
                with self.subTest(sidecar=name, panel=index):
                    self.assertGreater(panel["runtime_seconds"], 0.0)
