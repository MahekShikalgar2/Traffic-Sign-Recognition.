"""
SettingsPage — user-configurable application settings.
Saved to database/settings.json and applied live.
"""

import json
import logging
from pathlib import Path

import customtkinter as ctk

import src.config as _cfg
from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER, CONTENT,
    DATABASE_DIR,
)

logger = logging.getLogger(__name__)

SETTINGS_FILE = DATABASE_DIR / "settings.json"

_DEFAULTS = {
    "confidence_threshold": 0.60,
    "appearance_mode": "Dark",
    "font_size": "Normal",
    "max_history": 100,
    "show_confidence_warning": True,
}


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Fill any missing keys with defaults
            for k, v in _DEFAULTS.items():
                data.setdefault(k, v)
            return data
        except Exception as exc:
            logger.warning("Could not load settings.json: %s — using defaults.", exc)
    return dict(_DEFAULTS)


def _save_settings(settings: dict) -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


class SettingsPage(ctk.CTkFrame):
    """Application settings with live apply + save/reset."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._settings = _load_settings()
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(28, 0))
        ctk.CTkLabel(
            header, text="Settings",
            font=("Segoe UI", 24, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Configure application behaviour and appearance",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Scrollable content ───────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        scroll.grid(row=1, column=0, sticky="nsew", padx=30, pady=14)
        scroll.columnconfigure(0, weight=1)

        row_i = 0

        # ── Prediction settings ──────────────────────────────────────────
        row_i = self._section_header(scroll, row_i, "🔍 Prediction")

        # Confidence threshold
        thresh_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        thresh_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 10))
        thresh_card.columnconfigure(0, weight=1)
        row_i += 1

        thresh_top = ctk.CTkFrame(thresh_card, fg_color="transparent")
        thresh_top.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 6))
        thresh_top.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            thresh_top, text="Confidence Threshold",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            thresh_top,
            text="Predictions below this threshold are flagged as low-confidence.",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        self._thresh_val_lbl = ctk.CTkLabel(
            thresh_top,
            text=f"{self._settings['confidence_threshold']:.0%}",
            font=("Segoe UI", 14, "bold"), text_color=ACCENT,
        )
        self._thresh_val_lbl.grid(row=0, column=1, padx=(8, 0))

        self._thresh_slider = ctk.CTkSlider(
            thresh_card,
            from_=0.0, to=1.0, number_of_steps=20,
            fg_color=BORDER, progress_color=ACCENT,
            button_color=ACCENT, button_hover_color="#0EA5E9",
            command=self._on_threshold_change,
        )
        self._thresh_slider.set(self._settings["confidence_threshold"])
        self._thresh_slider.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")

        # Show confidence warning toggle
        warn_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        warn_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 10))
        warn_card.columnconfigure(0, weight=1)
        row_i += 1

        warn_inner = ctk.CTkFrame(warn_card, fg_color="transparent")
        warn_inner.grid(row=0, column=0, sticky="ew", padx=20, pady=16)
        warn_inner.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            warn_inner, text="Show Low-Confidence Warning",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            warn_inner, text="Display a warning banner when confidence is below threshold.",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        self._warn_switch = ctk.CTkSwitch(
            warn_inner, text="",
            fg_color=BORDER, progress_color=ACCENT,
            button_color=TEXT, button_hover_color=ACCENT,
            onvalue=True, offvalue=False,
        )
        self._warn_switch.grid(row=0, column=1, padx=(16, 0))
        if self._settings["show_confidence_warning"]:
            self._warn_switch.select()
        else:
            self._warn_switch.deselect()

        # ── Appearance settings ──────────────────────────────────────────
        row_i = self._section_header(scroll, row_i, "🎨 Appearance")

        appearance_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        appearance_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 10))
        appearance_card.columnconfigure(0, weight=1)
        row_i += 1

        app_inner = ctk.CTkFrame(appearance_card, fg_color="transparent")
        app_inner.grid(row=0, column=0, sticky="ew", padx=20, pady=16)
        app_inner.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            app_inner, text="Appearance Mode",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            app_inner, text="Switch between Dark, Light, or System default mode.",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        self._appearance_menu = ctk.CTkOptionMenu(
            app_inner,
            values=["Dark", "Light", "System"],
            font=("Segoe UI", 12),
            fg_color=CARD, button_color=ACCENT,
            button_hover_color="#0EA5E9",
            command=self._on_appearance_change,
        )
        self._appearance_menu.set(self._settings["appearance_mode"])
        self._appearance_menu.grid(row=0, column=1, padx=(16, 0))

        # Font size
        font_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        font_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 10))
        font_card.columnconfigure(0, weight=1)
        row_i += 1

        font_inner = ctk.CTkFrame(font_card, fg_color="transparent")
        font_inner.grid(row=0, column=0, sticky="ew", padx=20, pady=16)
        font_inner.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            font_inner, text="Font Size",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            font_inner, text="Adjust text size across the application.",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        self._font_menu = ctk.CTkOptionMenu(
            font_inner,
            values=["Small", "Normal", "Large"],
            font=("Segoe UI", 12),
            fg_color=CARD, button_color=ACCENT,
            button_hover_color="#0EA5E9",
        )
        self._font_menu.set(self._settings["font_size"])
        self._font_menu.grid(row=0, column=1, padx=(16, 0))

        # ── History settings ─────────────────────────────────────────────
        row_i = self._section_header(scroll, row_i, "📋 History")

        hist_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        hist_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 10))
        hist_card.columnconfigure(0, weight=1)
        row_i += 1

        hist_inner = ctk.CTkFrame(hist_card, fg_color="transparent")
        hist_inner.grid(row=0, column=0, sticky="ew", padx=20, pady=16)
        hist_inner.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hist_inner, text="Maximum History Records",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            hist_inner, text="Older predictions are removed when this limit is exceeded.",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        self._max_hist_var = ctk.StringVar(value=str(self._settings["max_history"]))
        self._max_hist_entry = ctk.CTkEntry(
            hist_inner, textvariable=self._max_hist_var,
            font=("Segoe UI", 12), width=80, height=32,
            fg_color=CARD, border_color=BORDER,
        )
        self._max_hist_entry.grid(row=0, column=1, padx=(16, 0))

        # ── Action buttons ───────────────────────────────────────────────
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.grid(row=row_i, column=0, sticky="ew", pady=(8, 20))
        row_i += 1

        ctk.CTkButton(
            btn_row,
            text="💾  Save Settings",
            font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT, hover_color="#0EA5E9", text_color="#0B1120",
            corner_radius=8, height=40, width=160,
            command=self._on_save,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(
            btn_row,
            text="↩  Reset Defaults",
            font=("Segoe UI", 13),
            fg_color=CARD, hover_color="#374151",
            border_width=1, border_color=BORDER,
            text_color=SUBTEXT, corner_radius=8, height=40, width=160,
            command=self._on_reset,
        ).pack(side="left")

        self._status_lbl = ctk.CTkLabel(
            btn_row, text="", font=("Segoe UI", 12), text_color=SUCCESS,
        )
        self._status_lbl.pack(side="left", padx=(16, 0))

    def _section_header(self, parent, row_i, title) -> int:
        ctk.CTkLabel(
            parent, text=title,
            font=("Segoe UI", 13, "bold"), text_color=SUBTEXT,
        ).grid(row=row_i, column=0, sticky="w", pady=(12, 6))
        return row_i + 1

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_threshold_change(self, value: float) -> None:
        self._thresh_val_lbl.configure(text=f"{value:.0%}")

    def _on_appearance_change(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)

    def _on_save(self) -> None:
        # Validate max_history
        try:
            max_hist = int(self._max_hist_var.get())
            if max_hist < 1 or max_hist > 10000:
                raise ValueError
        except ValueError:
            self._status_lbl.configure(
                text="⚠  Max History must be an integer between 1 and 10000.",
                text_color=WARNING,
            )
            return

        self._settings["confidence_threshold"] = round(self._thresh_slider.get(), 2)
        self._settings["appearance_mode"] = self._appearance_menu.get()
        self._settings["font_size"] = self._font_menu.get()
        self._settings["max_history"] = max_hist
        self._settings["show_confidence_warning"] = bool(self._warn_switch.get())

        # Apply to live config
        _cfg.CONFIDENCE_THRESHOLD = self._settings["confidence_threshold"]
        _cfg.MAX_HISTORY = max_hist

        try:
            _save_settings(self._settings)
            self._status_lbl.configure(text="✓  Settings saved.", text_color=SUCCESS)
            self.after(3000, lambda: self._status_lbl.configure(text=""))
        except Exception as exc:
            self._status_lbl.configure(
                text=f"Error saving: {exc}", text_color=ERROR,
            )

    def _on_reset(self) -> None:
        self._settings = dict(_DEFAULTS)
        self._thresh_slider.set(_DEFAULTS["confidence_threshold"])
        self._thresh_val_lbl.configure(
            text=f"{_DEFAULTS['confidence_threshold']:.0%}"
        )
        self._appearance_menu.set(_DEFAULTS["appearance_mode"])
        self._font_menu.set(_DEFAULTS["font_size"])
        self._max_hist_var.set(str(_DEFAULTS["max_history"]))
        if _DEFAULTS["show_confidence_warning"]:
            self._warn_switch.select()
        else:
            self._warn_switch.deselect()
        ctk.set_appearance_mode(_DEFAULTS["appearance_mode"])
        self._status_lbl.configure(text="↩  Defaults restored.", text_color=SUBTEXT)
        self.after(3000, lambda: self._status_lbl.configure(text=""))

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload settings from disk on each visit."""
        self._settings = _load_settings()
        self._thresh_slider.set(self._settings["confidence_threshold"])
        self._thresh_val_lbl.configure(
            text=f"{self._settings['confidence_threshold']:.0%}"
        )
