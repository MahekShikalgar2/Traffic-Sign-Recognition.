"""
run.py — application entry point for Traffic Sign Recognition.

IMPORTANT: os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' is set as the very
first statement so it takes effect before TensorFlow is ever imported.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # must be first

import sys
from pathlib import Path

# Ensure workspace root and src/ are on sys.path so all imports resolve
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Logging setup ────────────────────────────────────────────────────────────
from src.utils.logging_config import setup_logging
setup_logging()

import logging
logger = logging.getLogger(__name__)
logger.info("Traffic Sign Recognition application starting …")

# ── Service layer ────────────────────────────────────────────────────────────
from src.services.model_service import ModelService
from src.services.prediction_service import PredictionService
from src.services.image_service import ImageService
from src.services.history_service import HistoryService

model_service = ModelService()
logger.info("Loading CNN model …")
model_service.load()

prediction_service = PredictionService(model_service)
image_service = ImageService()
history_service = HistoryService()

# ── UI ───────────────────────────────────────────────────────────────────────
# Import the UI only after services are ready so the app never shows a blank
# window while the model loads.
try:
    from src.ui.app import App
    app = App(
        model_service=model_service,
        prediction_service=prediction_service,
        image_service=image_service,
        history_service=history_service,
    )
    app.mainloop()
except ImportError as exc:
    # FEAT-002 (UI) may not exist yet during FEAT-001 testing — that is fine.
    logger.warning("UI not available yet (%s). Running in headless mode.", exc)
    info = model_service.get_model_info()
    logger.info("Model info: %s", info)
    print("Services loaded successfully. UI not yet implemented.")
