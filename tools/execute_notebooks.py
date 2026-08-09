"""Execute every canonical notebook in a clean kernel and reject warnings."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "notebook"
NOTEBOOKS = [
    "figure1_2_3.ipynb",
    "figure4.ipynb",
    "figure5.ipynb",
    "PMGW_corrected.ipynb",
]


def execute(path):
    value = nbformat.read(path, as_version=4)
    client = NotebookClient(
        value,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    warning_streams = []
    for cell_index, cell in enumerate(value.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stderr":
                continue
            text = output.get("text", "")
            if "warning" in text.lower():
                warning_streams.append((cell_index, text.strip()))
    if warning_streams:
        details = "\n".join(
            f"cell {cell_index}: {stream}" for cell_index, stream in warning_streams
        )
        raise RuntimeError(f"{path.name} emitted warnings:\n{details}")
    nbformat.write(value, path)
    print(f"executed {path}")


def main():
    for name in NOTEBOOKS:
        execute(NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
