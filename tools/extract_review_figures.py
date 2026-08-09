"""Extract the seven canonical rendered PNGs for the issue #8 comparison."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "review" / "figures" / "after-issue6"
MAPPING = {
    "figure1_2_3.ipynb": ["figure01.png", "figure02.png", "figure03.png"],
    "figure4.ipynb": ["figure04.png"],
    "figure5.ipynb": ["figure05.png"],
    "PMGW_corrected.ipynb": ["figure06.png", "figure07.png"],
}


def image_outputs(notebook):
    for cell_index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if "image/png" in data:
                yield cell_index, output_index, base64.b64decode(data["image/png"])


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = []
    for notebook_name, output_names in MAPPING.items():
        notebook_path = ROOT / "docs" / "notebook" / notebook_name
        notebook = nbformat.read(notebook_path, as_version=4)
        images = list(image_outputs(notebook))
        if len(images) != len(output_names):
            raise RuntimeError(
                f"{notebook_name}: expected {len(output_names)} PNG outputs, "
                f"found {len(images)}"
            )
        for output_name, (cell_index, output_index, image) in zip(output_names, images):
            destination = DESTINATION / output_name
            destination.write_bytes(image)
            records.append(
                {
                    "figure": int(output_name[6:8]),
                    "file": output_name,
                    "notebook": f"docs/notebook/{notebook_name}",
                    "cell": cell_index,
                    "output": output_index,
                    "sha256": hashlib.sha256(image).hexdigest(),
                    "bytes": len(image),
                }
            )
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "source_commit": commit,
        "description": "Canonical post-issue-6 outputs extracted from clean notebook runs",
        "figures": records,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} figures to {DESTINATION}")


if __name__ == "__main__":
    main()
