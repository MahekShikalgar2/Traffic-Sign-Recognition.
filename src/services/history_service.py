"""
HistoryService — persists prediction records to database/history.json.
Thread-safe via threading.Lock.
"""

import csv
import json
import logging
import threading
from datetime import datetime, date
from pathlib import Path

import src.config as _cfg  # import the module so patching _cfg.MAX_HISTORY in tests works

logger = logging.getLogger(__name__)


class HistoryService:
    """
    Stores and retrieves prediction history as a JSON list.

    Schema of each record
    ---------------------
    {
        "id":                  str   (ISO-timestamp-based unique ID),
        "timestamp":           str   (ISO 8601),
        "filename":            str,
        "predicted_class":     int   (0–42),
        "predicted_label":     str,
        "confidence":          float (0–1),
        "top3":                list[{class_id, label, probability}],
        "processing_time_ms":  float
    }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ensure_db_dir()

    # Convenience accessors so the rest of the class always reads live config
    @staticmethod
    def _db_dir():
        return _cfg.DATABASE_DIR

    @staticmethod
    def _hist_file():
        return _cfg.HISTORY_FILE

    @staticmethod
    def _max_history():
        return _cfg.MAX_HISTORY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_record(self, record: dict) -> None:
        """
        Append *record* to history, trim to MAX_HISTORY newest, then save.
        """
        with self._lock:
            records = self._load()
            records.append(record)
            # Keep only the most recent MAX_HISTORY entries
            if len(records) > self._max_history():
                records = records[-self._max_history():]
            self._save(records)
            logger.debug("History record added (total=%d).", len(records))

    def get_all(self) -> list:
        """Return all records, newest first."""
        with self._lock:
            records = self._load()
        return list(reversed(records))

    def get_stats(self) -> dict:
        """
        Return aggregate statistics over all stored predictions.

        Returns
        -------
        dict with keys:
            total_predictions  int
            most_common_sign   str  (label, or '' if no records)
            avg_confidence     float
            today_count        int
        """
        with self._lock:
            records = self._load()

        if not records:
            return {
                "total_predictions": 0,
                "most_common_sign":  "",
                "avg_confidence":    0.0,
                "today_count":       0,
            }

        total = len(records)
        avg_conf = sum(r.get("confidence", 0.0) for r in records) / total

        # Most common sign
        from collections import Counter
        label_counts = Counter(r.get("predicted_label", "") for r in records)
        most_common_sign = label_counts.most_common(1)[0][0]

        # Today's count
        today_str = date.today().isoformat()
        today_count = sum(
            1 for r in records
            if r.get("timestamp", "").startswith(today_str)
        )

        return {
            "total_predictions": total,
            "most_common_sign":  most_common_sign,
            "avg_confidence":    round(avg_conf, 4),
            "today_count":       today_count,
        }

    def clear_history(self) -> None:
        """Delete all stored history records."""
        with self._lock:
            self._save([])
        logger.info("Prediction history cleared.")

    def export_csv(self, path: str) -> None:
        """
        Export all history records to a CSV file at *path*.
        """
        with self._lock:
            records = self._load()

        fieldnames = [
            "id", "timestamp", "filename", "predicted_class",
            "predicted_label", "confidence", "processing_time_ms",
        ]

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in reversed(records):   # newest first
                writer.writerow(r)

        logger.info("History exported to %s (%d records).", path, len(records))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_db_dir(self) -> None:
        self._db_dir().mkdir(parents=True, exist_ok=True)

    def _load(self) -> list:
        """Load and return the raw list from JSON (caller holds lock)."""
        hist_file = self._hist_file()
        if not hist_file.exists():
            return []
        try:
            with open(hist_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                logger.warning("history.json has unexpected structure — resetting.")
                return []
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load history.json: %s", exc)
            return []

    def _save(self, records: list) -> None:
        """Persist *records* to HISTORY_FILE (caller holds lock)."""
        self._ensure_db_dir()
        try:
            with open(self._hist_file(), "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Failed to save history.json: %s", exc)
