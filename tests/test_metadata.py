"""Regression tests for machine-readable figure provenance."""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pgot import write_figure_sidecar


class FigureSidecarSchemaTests(unittest.TestCase):
    def test_generation_time_is_distinct_from_reproducible_source_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "figure.png"
            sidecar = root / "figure.json"
            output.write_bytes(b"figure bytes")

            before = datetime.now(timezone.utc)
            with patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "946684800"}):
                payload = write_figure_sidecar(
                    sidecar,
                    figure=1,
                    outputs=[output],
                    panels=[],
                    inputs={},
                    seeds={},
                )
            after = datetime.now(timezone.utc)

        generated_at = datetime.fromisoformat(payload["generated_at"])
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["source_date_epoch"], 946684800)
        self.assertGreaterEqual(generated_at, before)
        self.assertLessEqual(generated_at, after)
        self.assertNotEqual(payload["generated_at"], "2000-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
