"""
Unit tests for src/services/prediction_service.py

Uses the real model to verify the full predict() pipeline produces a
correctly-typed result dict with valid values.
"""

import os
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.config import MODEL_PATH, NUM_CLASSES, CONFIDENCE_THRESHOLD


@unittest.skipUnless(MODEL_PATH.exists(), f"my_model.h5 not found at {MODEL_PATH}")
class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.services.model_service import ModelService
        from src.services.prediction_service import PredictionService

        cls.model_svc = ModelService()
        cls.model_svc.load()
        cls.pred_svc = PredictionService(cls.model_svc)

    def _make_test_image(self):
        """Return a small solid-grey PIL Image for smoke tests."""
        from PIL import Image
        return Image.new("RGB", (64, 64), color=(128, 128, 128))

    # ------------------------------------------------------------------ result keys

    def test_result_has_required_keys(self):
        img = self._make_test_image()
        result = self.pred_svc.predict(img)
        required = {
            "predicted_class", "predicted_label", "confidence",
            "top3", "is_low_confidence", "processing_time_ms",
        }
        self.assertTrue(required.issubset(result.keys()))

    # ------------------------------------------------------------------ types

    def test_predicted_class_is_int_in_range(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertIsInstance(result["predicted_class"], int)
        self.assertGreaterEqual(result["predicted_class"], 0)
        self.assertLessEqual(result["predicted_class"], NUM_CLASSES - 1)

    def test_predicted_label_is_string(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertIsInstance(result["predicted_label"], str)
        self.assertGreater(len(result["predicted_label"]), 0)

    def test_confidence_is_float_in_unit_interval(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_is_low_confidence_type(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertIsInstance(result["is_low_confidence"], bool)

    def test_processing_time_ms_positive(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertGreater(result["processing_time_ms"], 0.0)

    # ------------------------------------------------------------------ top3

    def test_top3_is_list_of_three(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertIsInstance(result["top3"], list)
        self.assertEqual(len(result["top3"]), 3)

    def test_top3_entries_have_correct_keys(self):
        result = self.pred_svc.predict(self._make_test_image())
        for entry in result["top3"]:
            self.assertIn("class_id", entry)
            self.assertIn("label", entry)
            self.assertIn("probability", entry)

    def test_top3_sorted_descending(self):
        result = self.pred_svc.predict(self._make_test_image())
        probs = [e["probability"] for e in result["top3"]]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top3_first_matches_prediction(self):
        result = self.pred_svc.predict(self._make_test_image())
        self.assertEqual(result["top3"][0]["class_id"], result["predicted_class"])

    # ------------------------------------------------------------------ label lookup

    def test_label_from_signs_py(self):
        """Label must come from signs.classes (keys 1-43, pred+1 lookup)."""
        import signs
        result = self.pred_svc.predict(self._make_test_image())
        expected_label = signs.classes[result["predicted_class"] + 1]
        self.assertEqual(result["predicted_label"], expected_label)

    # ------------------------------------------------------------------ grayscale input

    def test_grayscale_input_works(self):
        from PIL import Image
        grey_img = Image.new("L", (64, 64), color=100)
        result = self.pred_svc.predict(grey_img)
        self.assertIn("predicted_class", result)

    def test_rgba_input_works(self):
        from PIL import Image
        rgba_img = Image.new("RGBA", (64, 64), color=(200, 100, 50, 255))
        result = self.pred_svc.predict(rgba_img)
        self.assertIn("predicted_class", result)


if __name__ == "__main__":
    unittest.main()
