"""
Unit tests for src/services/history_service.py

Uses a temporary directory so the real database/history.json is never touched.
"""

import sys
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def _make_record(label: str = "Stop", confidence: float = 0.92,
                 today: bool = True) -> dict:
    """Return a minimal history record dict."""
    ts = datetime.now().isoformat() if today else "2020-01-01T00:00:00"
    return {
        "id":                 ts,
        "timestamp":          ts,
        "filename":           "test.jpg",
        "predicted_class":    14,
        "predicted_label":    label,
        "confidence":         confidence,
        "top3":               [{"class_id": 14, "label": label, "probability": confidence}],
        "processing_time_ms": 12.5,
    }


class TestHistoryService(unittest.TestCase):

    def setUp(self):
        """Redirect DATABASE_DIR and HISTORY_FILE to a temp directory."""
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)

        import src.config as cfg
        self._orig_db_dir = cfg.DATABASE_DIR
        self._orig_hist   = cfg.HISTORY_FILE

        cfg.DATABASE_DIR  = tmp_path
        cfg.HISTORY_FILE  = tmp_path / "history.json"

        # Re-import to pick up patched constants
        import importlib
        import src.services.history_service as hs_mod
        importlib.reload(hs_mod)
        from src.services.history_service import HistoryService
        self.svc = HistoryService()

    def tearDown(self):
        import src.config as cfg
        cfg.DATABASE_DIR = self._orig_db_dir
        cfg.HISTORY_FILE = self._orig_hist
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------ add / get

    def test_add_and_get_record(self):
        rec = _make_record()
        self.svc.add_record(rec)
        records = self.svc.get_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["predicted_label"], "Stop")

    def test_get_all_returns_newest_first(self):
        r1 = _make_record("Stop", 0.9)
        r1["timestamp"] = "2024-01-01T10:00:00"
        r2 = _make_record("Yield", 0.8)
        r2["timestamp"] = "2024-01-01T11:00:00"
        self.svc.add_record(r1)
        self.svc.add_record(r2)
        records = self.svc.get_all()
        self.assertEqual(records[0]["predicted_label"], "Yield")

    def test_empty_get_all(self):
        self.assertEqual(self.svc.get_all(), [])

    # ------------------------------------------------------------------ persistence

    def test_persists_to_json(self):
        import src.config as cfg
        rec = _make_record()
        self.svc.add_record(rec)
        with open(cfg.HISTORY_FILE, "r") as fh:
            data = json.load(fh)
        self.assertEqual(len(data), 1)

    # ------------------------------------------------------------------ max_history trimming

    def test_trims_to_max_history(self):
        import src.config as cfg
        orig_max = cfg.MAX_HISTORY
        cfg.MAX_HISTORY = 5
        try:
            for i in range(8):
                self.svc.add_record(_make_record(f"Sign{i}"))
            records = self.svc.get_all()
            self.assertLessEqual(len(records), 5)
        finally:
            cfg.MAX_HISTORY = orig_max

    # ------------------------------------------------------------------ clear

    def test_clear_history(self):
        self.svc.add_record(_make_record())
        self.svc.clear_history()
        self.assertEqual(self.svc.get_all(), [])

    # ------------------------------------------------------------------ stats

    def test_stats_empty(self):
        stats = self.svc.get_stats()
        self.assertEqual(stats["total_predictions"], 0)
        self.assertEqual(stats["most_common_sign"], "")
        self.assertEqual(stats["avg_confidence"], 0.0)
        self.assertEqual(stats["today_count"], 0)

    def test_stats_total_predictions(self):
        for _ in range(3):
            self.svc.add_record(_make_record())
        stats = self.svc.get_stats()
        self.assertEqual(stats["total_predictions"], 3)

    def test_stats_avg_confidence(self):
        self.svc.add_record(_make_record(confidence=0.8))
        self.svc.add_record(_make_record(confidence=0.6))
        stats = self.svc.get_stats()
        self.assertAlmostEqual(stats["avg_confidence"], 0.7, places=4)

    def test_stats_most_common_sign(self):
        self.svc.add_record(_make_record("Stop"))
        self.svc.add_record(_make_record("Stop"))
        self.svc.add_record(_make_record("Yield"))
        stats = self.svc.get_stats()
        self.assertEqual(stats["most_common_sign"], "Stop")

    def test_stats_today_count(self):
        self.svc.add_record(_make_record(today=True))
        self.svc.add_record(_make_record(today=True))
        self.svc.add_record(_make_record(today=False))
        stats = self.svc.get_stats()
        self.assertEqual(stats["today_count"], 2)

    # ------------------------------------------------------------------ export CSV

    def test_export_csv(self):
        import csv
        import src.config as cfg
        self.svc.add_record(_make_record("Stop", 0.9))
        self.svc.add_record(_make_record("Yield", 0.7))
        out_path = str(Path(self._tmpdir.name) / "export.csv")
        self.svc.export_csv(out_path)
        with open(out_path, newline="", encoding="utf-8") as fh:
            reader = list(csv.DictReader(fh))
        self.assertEqual(len(reader), 2)
        self.assertIn("predicted_label", reader[0])


if __name__ == "__main__":
    unittest.main()
