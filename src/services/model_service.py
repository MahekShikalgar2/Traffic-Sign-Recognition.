"""
ModelService — loads the trained Keras CNN model once and exposes predict().

IMPORTANT: os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' is set at module level,
before any TensorFlow / Keras symbol is imported, to suppress oneDNN noise.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # must appear before TF import

import time
import logging
from pathlib import Path

import numpy as np

from src.config import MODEL_PATH

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton-style service that owns the Keras model.

    Usage
    -----
    svc = ModelService()
    svc.load()                   # call once on the main thread at startup
    probs = svc.predict(arr)     # safe to call from a worker thread after load
    info  = svc.get_model_info()
    """

    def __init__(self) -> None:
        self._model = None
        self._model_path: Path = MODEL_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the Keras model from MODEL_PATH.
        Idempotent — does nothing if the model is already loaded.
        """
        if self._model is not None:
            logger.debug("ModelService.load() called but model is already loaded — skipping.")
            return

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                "Ensure my_model.h5 is in the workspace root."
            )

        logger.info("Loading model from %s …", self._model_path)
        t0 = time.perf_counter()

        # Defer the heavy Keras import to here so the env-var is already set
        from keras.models import load_model  # noqa: PLC0415
        self._model = load_model(str(self._model_path))

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2f s.", elapsed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run model.predict on pre-processed input array *x*.

        Parameters
        ----------
        x : np.ndarray
            Shape (1, 32, 32, 1), dtype float32, values in [0, 1].

        Returns
        -------
        np.ndarray
            Shape (1, 43) — raw softmax probabilities.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        return self._model.predict(x, verbose=0)

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded model.

        Returns
        -------
        dict with keys:
            input_shape, output_shape, num_layers, total_params,
            model_path, file_size_mb
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        input_shape = tuple(self._model.input_shape)
        output_shape = tuple(self._model.output_shape)
        num_layers = len(self._model.layers)
        total_params = self._model.count_params()
        file_size_mb = round(self._model_path.stat().st_size / (1024 * 1024), 2)

        return {
            "input_shape":   input_shape,
            "output_shape":  output_shape,
            "num_layers":    num_layers,
            "total_params":  total_params,
            "model_path":    str(self._model_path),
            "file_size_mb":  file_size_mb,
        }

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded."""
        return self._model is not None
