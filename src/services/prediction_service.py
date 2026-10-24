"""
PredictionService — wraps the model + signs dictionary to produce a
fully-typed prediction result dict from a PIL Image.

Key contract
------------
* signs.py keys are 1–43; model outputs class IDs 0–42.
  Always look up  signs.classes[predicted_class + 1].
* Preprocessing matches the training notebook exactly.
"""

import sys
import time
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from src.config import BASE_DIR, CONFIDENCE_THRESHOLD

# Make workspace root importable so that 'import signs' works
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import signs  # noqa: E402 — must come after sys.path fix

if TYPE_CHECKING:
    from src.services.model_service import ModelService

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Thin service that wraps a ModelService instance and exposes predict().

    Parameters
    ----------
    model_service : ModelService
        A loaded ModelService instance (model_service.load() already called).
    """

    def __init__(self, model_service: 'ModelService') -> None:
        self._model_service = model_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Run end-to-end prediction for a single PIL Image.

        Preprocessing pipeline (must match training notebook):
          1. Convert to grayscale  (.convert('L'))
          2. Resize to 32×32       (.resize((32,32), Image.LANCZOS))
          3. NumPy array float32   (np.array / 255.0)
          4. Add channel dim       (expand_dims axis=-1)  → (32,32,1)
          5. Add batch dim         (expand_dims axis=0)   → (1,32,32,1)
          6. model.predict         → (1,43) probabilities
          7. argmax                → predicted class 0–42
          8. label lookup          → signs.classes[pred + 1]

        Returns
        -------
        dict with keys:
            predicted_class   int       0–42
            predicted_label   str       from signs.py
            confidence        float     0–1
            top3              list[dict] [{class_id, label, probability}]
            is_low_confidence bool
            processing_time_ms float
        """
        t0 = time.perf_counter()

        # ---- 1. Grayscale ------------------------------------------------
        img = pil_image.convert('L')

        # ---- 2. Resize ---------------------------------------------------
        img = img.resize((32, 32), Image.LANCZOS)

        # ---- 3. NumPy + normalise ----------------------------------------
        arr = np.array(img, dtype=np.float32) / 255.0

        # ---- 4. Channel dim  (32,32) → (32,32,1) -------------------------
        arr = np.expand_dims(arr, axis=-1)

        # ---- 5. Batch dim   (32,32,1) → (1,32,32,1) ----------------------
        arr = np.expand_dims(arr, axis=0)

        # ---- 6. Predict --------------------------------------------------
        raw_probs = self._model_service.predict(arr)   # shape (1, 43)
        probs = raw_probs[0]                           # shape (43,)

        # ---- 7. Class & confidence ---------------------------------------
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # ---- 8. Label lookup  (signs keys are 1-indexed) -----------------
        predicted_label = signs.classes.get(pred_class + 1, f"Unknown class {pred_class}")

        # ---- Top 3 -------------------------------------------------------
        sorted_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "class_id":    int(idx),
                "label":       signs.classes.get(int(idx) + 1, f"Unknown class {idx}"),
                "probability": float(probs[idx]),
            }
            for idx in sorted_indices
        ]

        processing_time_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "Prediction: class=%d label='%s' confidence=%.4f time=%.1f ms",
            pred_class, predicted_label, confidence, processing_time_ms,
        )

        return {
            "predicted_class":    pred_class,
            "predicted_label":    predicted_label,
            "confidence":         confidence,
            "top3":               top3,
            "is_low_confidence":  confidence < CONFIDENCE_THRESHOLD,
            "processing_time_ms": processing_time_ms,
        }
