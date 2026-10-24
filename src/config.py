"""
Central configuration module for Traffic Sign Recognition application.
All paths, constants, and colour tokens live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent          # workspace root
MODEL_PATH = BASE_DIR / 'my_model.h5'
SIGNS_PATH = BASE_DIR / 'signs.py'
DATASET_DIR = BASE_DIR / 'datasettraffic'
DATABASE_DIR = BASE_DIR / 'database'
HISTORY_FILE = DATABASE_DIR / 'history.json'
LOG_DIR = BASE_DIR / 'logs'

# ---------------------------------------------------------------------------
# Model / image constants
# ---------------------------------------------------------------------------
IMG_SIZE = (32, 32)
NUM_CLASSES = 43
CONFIDENCE_THRESHOLD = 0.60   # default — can be overridden from Settings page
MAX_HISTORY = 100
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')
MAX_FILE_SIZE_MB = 10

# ---------------------------------------------------------------------------
# Dataset class counts  (keys 0–42, verified sum == 34 776)
# ---------------------------------------------------------------------------
DATASET_CLASS_COUNTS = {
    0:   157,
    1:  1980,
    2:  2010,
    3:  1260,
    4:  1770,
    5:  1650,
    6:   360,
    7:  1290,
    8:  1260,
    9:  1320,
    10: 1800,
    11: 1170,
    12: 1890,
    13: 1920,
    14:  690,
    15:  540,
    16:  360,
    17:  990,
    18: 1080,
    19:  180,
    20:  300,
    21:  270,
    22:  330,
    23:  450,
    24:  240,
    25: 1350,
    26:  540,
    27:  210,
    28:  480,
    29:  240,
    30:  390,
    31:  690,
    32:  210,
    33:  599,
    34:  360,
    35: 1080,
    36:  330,
    37:  180,
    38: 1860,
    39:  270,
    40:  300,
    41:  210,
    42:  210,
}

assert sum(DATASET_CLASS_COUNTS.values()) == 34776, (
    f"DATASET_CLASS_COUNTS sum mismatch: {sum(DATASET_CLASS_COUNTS.values())}"
)

# ---------------------------------------------------------------------------
# UI colour palette (dark AI-dashboard theme)
# ---------------------------------------------------------------------------
BG        = '#0B1120'
SIDEBAR   = '#111827'
CONTENT   = '#151E2E'
CARD      = '#1E293B'
ACCENT    = '#38BDF8'
ACCENT2   = '#8B5CF6'
SUCCESS   = '#22C55E'
WARNING   = '#F59E0B'
ERROR     = '#EF4444'
TEXT      = '#F8FAFC'
SUBTEXT   = '#94A3B8'
BORDER    = '#2D3748'

# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------
WINDOW_W = 1280
WINDOW_H = 800
MIN_W    = 1024
MIN_H    = 700
