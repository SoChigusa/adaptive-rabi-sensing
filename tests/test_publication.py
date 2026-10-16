"""Protect the cached-data workflow against missing or corrupted inputs."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce


class PublicationTests(unittest.TestCase):
    def test_input_integrity_before_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = b"published input"
            (root / "sample.npz").write_bytes(sample)
            (root / "manifest.json").write_text(json.dumps({"inputs": [{
                "file": "sample.npz", "sha256": hashlib.sha256(sample).hexdigest(),
            }]}))
            with patch.object(reproduce, "ROOT", root):
                reproduce.verify_inputs()
                (root / "sample.npz").write_bytes(b"altered input")
                with self.assertRaises(RuntimeError):
                    reproduce.verify_inputs()
                (root / "sample.npz").unlink()
                with self.assertRaises(RuntimeError):
                    reproduce.verify_inputs()
