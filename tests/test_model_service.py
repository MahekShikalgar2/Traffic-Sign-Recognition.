"""
Unit tests for src/services/model_service.py

These tests load the real my_model.h5 and verify shape + basic predict().
They are skipped automatically if the model file is absent (CI without model).
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure workspace root is importable
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.config import MODEL_PATH, IMG_SIZE, NUM_CLASSES


@unittest.skipUnless(MODEL_PATH.exists(), f"my_model.h5 not found at {MODEL_PATH}")
class TestModelService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.services.model_service import ModelService
        cls.svc = ModelService()
        cls.svc.load()

    def test_load_is_idempotent(self):
        """Calling load() twice should not raise."""
        self.svc.load()  # second call — must be a no-op
        self.assertTrue(self.svc.is_loaded)

    def test_get_model_info_input_shape(self):
        info = self.svc.get_model_info()
        self.assertIn("input_shape", info)
        # Expect (None, 32, 32, 1)
        self.assertEqual(info["input_shape"], (None, IMG_SIZE[0], IMG_SIZE[1], 1))

    def test_get_model_info_output_shape(self):
        info = self.svc.get_model_info()
        self.assertIn("output_shape", info)
        # Expect (None, 43)
        self.assertEqual(info["output_shape"], (None, NUM_CLASSES))

    def test_get_model_info_keys(self):
        info = self.svc.get_model_info()
        expected_keys = {
            "input_shape", "output_shape", "num_layers",
            "total_params", "model_path", "file_size_mb",
        }
        self.assertTrue(expected_keys.issubset(info.keys()))

    def test_predict_output_shape(self):
        import numpy as np
        dummy = np.zeros((1, 32, 32, 1), dtype=np.float32)
        result = self.svc.predict(dummy)
        self.assertEqual(result.shape, (1, NUM_CLASSES))

    def test_predict_probabilities_sum_to_one(self):
        import numpy as np
        dummy = np.zeros((1, 32, 32, 1), dtype=np.float32)
        result = self.svc.predict(dummy)
        self.assertAlmostEqual(float(result[0].sum()), 1.0, places=4)

    def test_model_not_loaded_raises(self):
        from src.services.model_service import ModelService
        fresh = ModelService()
        with self.assertRaises(RuntimeError):
            fresh.predict(None)

    def test_missing_model_raises(self):
        from src.services.model_service import ModelService
        svc = ModelService()
        svc._model_path = Path("/nonexistent/path/my_model.h5")
        with self.assertRaises(FileNotFoundError):
            svc.load()


if __name__ == "__main__":
    unittest.main()
