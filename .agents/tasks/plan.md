# Implementation Plan — Traffic Sign Recognition Professional Upgrade

## Exploration Findings

### Verified Facts
- **Python**: `C:\Users\Mahek\AppData\Local\Programs\Python\Python311\python.exe` (3.11.9) — use this always
- **Model**: `my_model.h5` loads cleanly with `from keras.models import load_model` (keras 2.15.0, TF 2.15.1)
  - Input: `(None, 32, 32, 1)` grayscale
  - Output: `(None, 43)` softmax — 12 layers, 281,153 trainable params
- **signs.py**: Keys **1–43** → lookup key = `predicted_class_id + 1`
- **Training preprocessing** (from notebook cell 6): `cv2.cvtColor(BGR→GRAY)` + `/255` → reshape to `(N,32,32,1)`
  - PIL equivalent (used in Front-end.py and correct for inference): `.convert('L')` → `.resize((32,32))` → `np.array(float32)/255.0` → `expand_dims(-1)` → `expand_dims(0)`
- **Dataset**: 43 folders (0–42), 34,776 total images, min 157 / max 2,010 per class
- **Packages installed** under Python 3.11: tensorflow==2.15.1, keras==2.15.0, numpy==1.26.4, pillow==10.4.0, opencv-python==4.12.0.88, customtkinter==5.2.2 ✓, matplotlib==3.8.4 ✓
- **venv** at project root has only pip/setuptools — do not use it; use system Python 3.11 above
- **tkinter** and **matplotlib.backends.backend_tkagg.FigureCanvasTkAgg** both work correctly
- **CTk widgets confirmed**: CTk, CTkFrame, CTkButton, CTkLabel, CTkScrollableFrame, CTkProgressBar, CTkSlider, CTkEntry, CTkTextbox, CTkTabview, CTkSwitch, CTkOptionMenu, CTkComboBox, CTkCheckBox, set_appearance_mode, set_default_color_theme

### Detected Problems in Front-end.py
1. No normalization (`/255`) — preprocessing incomplete (fixed in new code)
2. No error handling — bare `except: pass` hides all errors
3. Loads model at module level with no env-var suppression — TF noise pollutes stdout
4. No confidence score, no top-3, no history, no result card
5. Window fixed at 800×600 with no minimum size — layout breaks on resize
6. Buttons recreated on every upload without cleanup
7. No thread safety — `model.predict()` blocks the UI thread

### Compatibility Notes
- `FigureCanvasTkAgg(fig, master=ctk_frame)` works — CTkFrame is a valid Tk master
- Do NOT mix `tk.ttk` styling with CTk; use CTk exclusively
- CTk 5.2.2 does not support `.pack_forget()` on grid-managed widgets — use `.grid_remove()` / `.grid()` for page switching
- `CTkProgressBar` in indeterminate mode: call `.start()` / `.stop()` (not `.configure(mode='indeterminate')` — set mode at construction)
- `set_appearance_mode` must be called before the first `CTk()` window is created, or use it live (CTk handles live changes)
- Background thread for prediction: use `threading.Thread`; dispatch result back via `root.after(0, callback)` — never update widgets from background thread directly

---

## Target File Structure

```
Traffic-Sign-Recognition/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── sign_info.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py
│   │   ├── prediction_service.py
│   │   ├── image_service.py
│   │   └── history_service.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── dashboard_page.py
│   │   ├── predict_page.py
│   │   ├── history_page.py
│   │   ├── dataset_page.py
│   │   ├── model_page.py
│   │   ├── settings_page.py
│   │   └── about_page.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       └── validators.py
├── database/             (runtime, gitignored)
├── assets/icons/         (no external files needed)
├── logs/                 (runtime, gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_model_service.py
│   ├── test_prediction_service.py
│   └── test_history_service.py
├── run.py
└── requirements.txt
```

---

## Implementation Plan

- [ ] 1. **Create directory skeleton and package init files**
      Create all directories: `src/`, `src/services/`, `src/ui/`, `src/utils/`, `database/`, `assets/icons/`, `logs/`, `tests/`. Create empty `__init__.py` in `src/`, `src/services/`, `src/ui/`, `src/utils/`, `tests/`.
      Files: `src/__init__.py`, `src/services/__init__.py`, `src/ui/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`
      Verify: `python -c "import src; import src.services; import src.ui; import src.utils; print('OK')"` prints OK.

- [ ] 2. **Create `src/config.py` — all paths, constants, colors**
      `BASE_DIR = Path(__file__).parent.parent` (workspace root). Defines: `MODEL_PATH`, `SIGNS_PATH`, `DATASET_DIR`, `DATABASE_DIR`, `HISTORY_FILE`, `SETTINGS_FILE`, `LOG_DIR`. Constants: `IMG_SIZE=(32,32)`, `NUM_CLASSES=43`, `DEFAULT_CONFIDENCE_THRESHOLD=0.60`, `MAX_HISTORY=100`, `SUPPORTED_FORMATS=('.jpg','.jpeg','.png','.bmp')`, `MAX_FILE_SIZE_MB=10`. Color scheme: `COLOR_BG='#0B1120'`, `COLOR_SIDEBAR='#111827'`, `COLOR_CONTENT='#151E2E'`, `COLOR_CARD='#1E293B'`, `COLOR_ACCENT='#38BDF8'`, `COLOR_ACCENT2='#8B5CF6'`, `COLOR_SUCCESS='#22C55E'`, `COLOR_WARNING='#F59E0B'`, `COLOR_ERROR='#EF4444'`, `COLOR_TEXT='#F8FAFC'`, `COLOR_SUBTEXT='#94A3B8'`. Window: `WINDOW_W=1280`, `WINDOW_H=800`, `MIN_W=1024`, `MIN_H=700`. `DATASET_CLASS_COUNTS` dict (keys 0–42 with actual image counts scanned from `datasettraffic/` at import time using `os.scandir` — cache result, total must equal 34,776).
      Files: `src/config.py`
      Verify: `python -c "from src.config import MODEL_PATH, COLOR_ACCENT, IMG_SIZE; assert IMG_SIZE==(32,32); print('config OK')"`.

- [ ] 3. **Create `src/utils/logging_config.py` and `src/utils/validators.py`**
      `logging_config.py`: `setup_logging()` sets up RotatingFileHandler (`logs/app.log`, 2MB, 3 backups) + StreamHandler (WARNING level for console). `get_logger(name)` returns `logging.getLogger(name)`. Calls `logging.captureWarnings(True)`.
      `validators.py`: `ImageValidator` with `validate_file(path) -> (bool, str)` and `validate_image(pil_image) -> (bool, str)`. Checks: file exists, extension in SUPPORTED_FORMATS, size ≤ MAX_FILE_SIZE_MB, non-zero bytes. Image checks: not None, mode valid, size > (0,0).
      Files: `src/utils/logging_config.py`, `src/utils/validators.py`
      Verify: `python -c "from src.utils.logging_config import get_logger; from src.utils.validators import ImageValidator; print('utils OK')"`.

- [ ] 4. **Create `src/services/model_service.py`**
      Set `os.environ['TF_ENABLE_ONEDNN_OPTS']='0'` at module top before keras import. `ModelService` class: `__init__` stores model path, `load()` calls `load_model(str(MODEL_PATH))`, logs load time; idempotent (skips if already loaded). `predict(x: np.ndarray) -> np.ndarray` calls `self._model.predict(x, verbose=0)`. `get_model_info() -> dict` returns `{input_shape, output_shape, num_layers, total_params, trainable_params, model_path, file_size_mb, layers: [{name, type, output_shape, params}]}` — extracts layer info from `model.layers` at load time. Thread-safety note: load on main thread; predict from background thread is safe for TF 2.15.
      Files: `src/services/model_service.py`
      Verify: `python -c "from src.services.model_service import ModelService; ms=ModelService(); ms.load(); info=ms.get_model_info(); assert info['input_shape']==(None,32,32,1); print('model_service OK')"`.

- [ ] 5. **Create `src/services/prediction_service.py`**
      `PredictionService(model_service: ModelService)`. Inserts `BASE_DIR` into `sys.path` then `from signs import classes as SIGN_CLASSES`. Key method `predict(pil_image: Image) -> dict`: pipeline: `.convert('L')` → `.resize((32,32), Image.LANCZOS)` → `np.array(float32)/255.0` → `expand_dims(-1)` → `expand_dims(0)` → `model_service.predict(arr)[0]` → `np.argmax` → label lookup `SIGN_CLASSES[pred+1]` → top-3 sorted descending. Returns `{predicted_class: int, predicted_label: str, confidence: float, top3: [{class_id,label,probability}×3], is_low_confidence: bool, processing_time_ms: float}`. Raises `ValueError` on None input.
      Files: `src/services/prediction_service.py`
      Verify: `python -c "from src.services.model_service import ModelService; from src.services.prediction_service import PredictionService; from PIL import Image; ms=ModelService(); ms.load(); ps=PredictionService(ms); r=ps.predict(Image.open(r'Output Images/o4.jpeg')); assert 0<=r['predicted_class']<=42; print('prediction_service OK, predicted:', r['predicted_label'])"`.

- [ ] 6. **Create `src/services/image_service.py` and `src/services/history_service.py`**
      `ImageService`: `load_image(path) -> Image` (try/except, friendly ValueError), `get_image_info(path, img) -> dict` (filename, filepath, dimensions, file_size_kb, mode, format), `resize_for_display(img, max_size=(400,400)) -> Image` (thumbnail, maintain aspect).
      `HistoryService`: `add_record(record)`, `get_all() -> list` (newest first), `get_stats() -> dict` (total_predictions, most_common_sign, avg_confidence, today_count), `clear_history()`, `export_csv(path)`. JSON persistence to `database/history.json`. Creates `database/` if missing. Thread-safe via `threading.Lock`. Each record has keys: `id`, `timestamp`, `filename`, `predicted_class`, `predicted_label`, `confidence`, `top3`, `processing_time_ms`.
      Files: `src/services/image_service.py`, `src/services/history_service.py`
      Verify: `python -c "from src.services.image_service import ImageService; from src.services.history_service import HistoryService; print('services OK')"`.

- [ ] 7. **Create `src/sign_info.py` — 43-entry sign information dictionary**
      `SIGN_INFO` dict keyed by class_id (0–42). Each value: `{name, category, meaning, driving_tip}`. Categories: Speed Limits (0–8), Prohibitory (9–15), Warning (16–30), Mandatory (31–38), Other (39–42). Populate all 43 entries with accurate, plain-English descriptions.
      Files: `src/sign_info.py`
      Verify: `python -c "from src.sign_info import SIGN_INFO; assert len(SIGN_INFO)==43; print('sign_info OK, sample:', SIGN_INFO[0]['name'])"`.

- [ ] 8. **Create `run.py` and `requirements.txt`**
      `run.py`: First line sets `os.environ['TF_ENABLE_ONEDNN_OPTS']='0'`. Then `sys.path` manipulation (adds workspace root and `src/`). Calls `setup_logging()`. Instantiates `ModelService`, calls `ms.load()` (blocking, on main thread). Imports and creates `App(model_service=ms)`. Calls `app.mainloop()`. `requirements.txt` pins: tensorflow==2.15.1, keras==2.15.0, customtkinter==5.2.2, matplotlib==3.8.4, Pillow==10.4.0, numpy==1.26.4, opencv-python==4.12.0.88.
      Files: `run.py`, `requirements.txt`
      Verify: `python -c "import ast, sys; ast.parse(open('run.py').read()); print('run.py syntax OK')"`.

- [ ] 9. **Create `src/ui/app.py` — App shell and sidebar**
      `App(ctk.CTk)`: calls `set_appearance_mode('dark')`, `set_default_color_theme('blue')` before `super().__init__()`. Title, geometry, minsize. Grid layout: column 0 (sidebar, 200px fixed, `minsize=200`), column 1 (content, weight=1). Sidebar `CTkFrame` with app logo label (`🚦 TSR`), subtitle, 7 nav `CTkButton`s (anchor=`'w'`, transparent bg, Unicode icons + label text), version label at bottom. Content area hosts all 7 page frames stacked in the same `grid(row=0, col=1)` cell; `show_page(name)` calls `.grid_remove()` on current page and `.grid()` on target page, then calls `page.refresh()`. `set_active_nav(name)` updates button fg_color and text_color. Constructor accepts `model_service`, `prediction_service`, `image_service`, `history_service` kwargs (all instantiated in `run.py` and passed in). Pages instantiated lazily or eagerly — eagerly is simpler and avoids first-click lag.
      Files: `src/ui/app.py`
      Verify: `python -c "import os; os.chdir(r'c:\\Users\\Mahek\\Traffic-Sign-Recognition'); import src.ui.app; print('app.py imports OK')"`.

- [ ] 10. **Create `src/ui/dashboard_page.py`**
       `DashboardPage(CTkFrame)`. Header: app title + subtitle. 4 stat cards in a row (`CTkFrame` bg=`COLOR_CARD`): Total Predictions (live from `history_service.get_stats()`), Dataset Size `34,776 images`, Classes `43 signs`, Model `~96% training accuracy`. Quick-action buttons: `Upload & Predict` → `app.show_page('predict')`, `View History` → `app.show_page('history')`. Recent predictions section: `CTkScrollableFrame`, last 5 records from `history_service.get_all()[:5]` as compact rows (timestamp, sign name, confidence badge colored per threshold). `refresh()` updates stat cards and recent list from history.
       Files: `src/ui/dashboard_page.py`
       Verify: import check + visual launch.

- [ ] 11. **Create `src/ui/predict_page.py` — core prediction UI**
       `PredictPage(CTkFrame)`. Two-column layout: left panel (upload, 480px min), right panel (results, fills rest). LEFT: dashed-border upload zone (`CTkFrame` with nested frames for visual border effect), `Upload Image` `CTkButton` (accent color), `Clear` secondary button. File info row (filename, dimensions, size) — shown only after load. `CTkLabel` image preview (max 380×300, aspect preserved). `Predict Sign` full-width `CTkButton` — `state='disabled'` until valid image loaded. RIGHT: placeholder label initially. After prediction: result `CTkFrame` (bg=`COLOR_CARD`): sign name (large font, white), class ID badge, confidence `CTkProgressBar` + percentage label colored green/amber/red by threshold, low-confidence warning `CTkLabel` (amber, visible only when `is_low_confidence`), top-3 rows each with sign name + `CTkProgressBar` + percentage, timestamp + processing time footer. Sign info section: `CTkFrame` showing category, meaning, driving_tip from `SIGN_INFO[predicted_class]`. Prediction runs in `threading.Thread`; indeterminate `CTkProgressBar` shown during inference; result dispatched via `self.after(0, _display_result)`. Error states shown as red text in result panel. Drag-and-drop: guarded `try/import tkinterdnd2` — silently skipped if not installed.
       Files: `src/ui/predict_page.py`
       Verify: import check + full manual test (upload image → predict → verify result displays correctly).

- [ ] 12. **Create `src/ui/history_page.py`**
       `HistoryPage(CTkFrame)`. Header row: title + `Export CSV` button + `Clear History` button (confirms with `CTkToplevel` dialog before clearing). Search `CTkEntry` filters list by sign name (case-insensitive substring). `CTkScrollableFrame` list: each row is a `CTkFrame` showing index, sign name, class ID, confidence badge, timestamp — clicking a row shows detail panel. Detail panel (right, `CTkFrame`): full record details. `refresh()` reloads from `history_service.get_all()`. Empty-state label when no records. Export calls `history_service.export_csv()` then `filedialog.asksaveasfilename`.
       Files: `src/ui/history_page.py`
       Verify: import check + manual test.

- [ ] 13. **Create `src/ui/dataset_page.py`**
       `DatasetPage(CTkFrame)`. Stats cards row. Matplotlib bar chart (all 43 classes × image count) embedded via `FigureCanvasTkAgg(fig, master=chart_frame)`. Chart styled: `fig.patch.set_facecolor('#151E2E')`, `ax.set_facecolor('#1E293B')`, bars `#38BDF8`, tick labels white, title white. Data sourced from `DATASET_CLASS_COUNTS` in config (or scanned dynamically if config dict sums ≠ 34,776). Class browser: `CTkScrollableFrame` listing all 43 signs from `signs.classes` with class ID and sign name. `refresh()` is a no-op (static data).
       Files: `src/ui/dataset_page.py`
       Verify: import check; no matplotlib `UserWarning` about non-GUI backend — use `matplotlib.use('TkAgg')` at import if needed.

- [ ] 14. **Create `src/ui/model_page.py`**
       `ModelPage(CTkFrame)`. Info cards: input shape, output shape, total params `281,153`, file size, num layers `12`, framework `TensorFlow 2.15.1 / Keras 2.15.0`. Architecture table in `CTkScrollableFrame`: 12 rows (one per layer) with columns name, type, output shape, params from `model_service.get_model_info()['layers']`. Training info section: dataset GTSRB, epochs 15, optimizer Adam lr=0.001, loss categorical_crossentropy, preprocessing pipeline description. Accuracy note: `~96% on training split — not guaranteed on real-world images`. `refresh()` is a no-op.
       Files: `src/ui/model_page.py`
       Verify: import check.

- [ ] 15. **Create `src/ui/settings_page.py`**
       `SettingsPage(CTkFrame)`. Loads/saves `database/settings.json`. Controls: Confidence Threshold `CTkSlider` (0.0–1.0, step 0.01) with live percentage label. Appearance Mode `CTkOptionMenu` (`['Dark','Light','System']`) calling `ctk.set_appearance_mode`. Max History `CTkEntry` (integer validation). Show low-confidence warning `CTkSwitch`. `Save Settings` button: writes JSON and updates `prediction_service.confidence_threshold`. `Reset Defaults` button. `refresh()` reloads settings file and updates widget states.
       Files: `src/ui/settings_page.py`
       Verify: import check + save/reload round-trip.

- [ ] 16. **Create `src/ui/about_page.py`**
       `AboutPage(CTkFrame)` in a `CTkScrollableFrame`. Sections: app logo + title (large); version `v1.0.0`; description paragraph; tech stack cards (Python 3.11, TF 2.15.1, Keras 2.15.0, CTk 5.2.2, NumPy 1.26.4, Pillow 10.4.0); GTSRB dataset citation; contributors (from README); disclaimer (`For educational purposes only — do not use for real-world safety decisions`). `refresh()` is a no-op.
       Files: `src/ui/about_page.py`
       Verify: import check.

- [ ] 17. **Create unit tests**
       `tests/test_model_service.py`: mock-based tests (singleton, get_model_info keys, load called once) + integration test `@unittest.skipUnless(MODEL_PATH.exists(), 'no model file')` that loads real model, predicts on dummy `(1,32,32,1)` float32 array, checks output shape `(1,43)` and `np.sum(probs) ≈ 1.0`.
       `tests/test_prediction_service.py`: loads real model, predicts on `Output Images/o4.jpeg`, asserts all required keys present, class in 0–42, confidence in [0,1], top3 sorted descending, label matches `signs.classes[pred+1]`.
       `tests/test_history_service.py`: all CRUD operations with temp dir, newest-first ordering, MAX_HISTORY trimming, clear, get_stats keys, export CSV headers.
       Files: `tests/test_model_service.py`, `tests/test_prediction_service.py`, `tests/test_history_service.py`
       Verify: `C:\Users\Mahek\AppData\Local\Programs\Python\Python311\python.exe -m unittest discover -s tests -p "test_*.py" -v` — 0 failures, 0 errors.

---

## Key Gotchas for the Implementer

1. **Python interpreter**: Always use `C:\Users\Mahek\AppData\Local\Programs\Python\Python311\python.exe` — the workspace venv and `py.exe` default to Python 3.14 which has no TF.
2. **TF env var**: `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'` must be the very first statement in `run.py` and in `model_service.py` (module level) — before any tensorflow/keras import anywhere in the process.
3. **Class ID offset**: `signs.classes` keys are 1–43; model outputs 0–42. Lookup is always `signs.classes[predicted_class + 1]`. Never shift the other way.
4. **Normalization**: Divide by 255.0 before predict. The training notebook normalizes (`img/255`) — omitting this degrades accuracy. The old `Front-end.py` was missing this step (confirmed by code inspection).
5. **Page switching**: Use `.grid_remove()` / `.grid()` — not `.pack_forget()` / `.pack()`. All pages live in the same grid cell `(row=0, col=1)`.
6. **Background thread**: `model.predict()` must run in `threading.Thread`; use `self.after(0, fn)` to push results back to the main thread for any CTk widget update.
7. **CTkProgressBar indeterminate**: Pass `mode='indeterminate'` at construction, then call `.start()` / `.stop()`. Do not call `.set()` on an indeterminate bar.
8. **matplotlib in CTk**: Set `matplotlib.use('TkAgg')` before importing `pyplot` (do it at the top of `dataset_page.py`). Pass the `CTkFrame` instance as `master` to `FigureCanvasTkAgg` — it works because CTkFrame is a valid Tk widget.
9. **DATASET_CLASS_COUNTS**: Verify sum == 34,776. If hardcoding fails, scan `datasettraffic/` once at startup and cache.
10. **run.py `sys.path`**: Insert `str(BASE_DIR)` (workspace root) so that `from signs import classes` works inside `prediction_service.py`, AND insert `str(BASE_DIR / 'src')` so relative imports within `src/` work when running `run.py` directly.
