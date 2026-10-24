"""
App — main CustomTkinter application window.

Architecture
------------
* App(ctk.CTk) is the root window.
* Services (model_service, prediction_service, image_service, history_service)
  are injected from run.py.
* All 7 pages are instantiated in __init__ and placed in grid(row=0, col=1)
  of the content area.
* show_page(name) grid_removes the current page, grids the target, then calls
  page.refresh().

IMPORTANT: ctk.set_appearance_mode / set_default_color_theme must be called
BEFORE ctk.CTk() is instantiated.  Those calls are made at module import time
so they take effect regardless of how App is constructed.
"""

import customtkinter as ctk

# ── These two calls must happen before any CTk widget is created ──────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
# ─────────────────────────────────────────────────────────────────────────

from src.config import (
    BG, SIDEBAR, CONTENT, CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR,
    TEXT, SUBTEXT, BORDER,
    WINDOW_W, WINDOW_H, MIN_W, MIN_H,
)

from src.ui.dashboard_page import DashboardPage
from src.ui.predict_page   import PredictPage
from src.ui.history_page   import HistoryPage
from src.ui.dataset_page   import DatasetPage
from src.ui.model_page     import ModelPage
from src.ui.settings_page  import SettingsPage
from src.ui.about_page     import AboutPage


# Nav item: (page_key, display_text, icon_char)
_NAV_ITEMS = [
    ("dashboard", "Dashboard",   "⊞"),
    ("predict",   "Predict Sign","◎"),
    ("history",   "History",     "📋"),
    ("dataset",   "Dataset",     "📊"),
    ("model",     "Model Info",  "🧠"),
    ("settings",  "Settings",    "⚙"),
    ("about",     "About",       "ℹ"),
]


class App(ctk.CTk):
    """Main application window."""

    def __init__(self, model_service, prediction_service,
                 image_service, history_service):
        super().__init__()

        # ── Window setup ────────────────────────────────────────────────
        self.title("Traffic Sign Recognition | AI Dashboard")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(MIN_W, MIN_H)
        self.configure(fg_color=BG)

        # ── Root grid: sidebar (col 0 fixed) + content (col 1 fills) ───
        self.columnconfigure(0, weight=0, minsize=200)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Sidebar ─────────────────────────────────────────────────────
        self._sidebar = ctk.CTkFrame(self, width=200, fg_color=SIDEBAR,
                                     corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.columnconfigure(0, weight=1)
        self._sidebar.rowconfigure(8, weight=1)   # pushes version to bottom
        self._sidebar.grid_propagate(False)

        # Logo / title
        logo_frame = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(22, 6))
        ctk.CTkLabel(
            logo_frame, text="🚦 TSR",
            font=("Segoe UI", 18, "bold"), text_color=ACCENT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            logo_frame, text="AI Dashboard",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w")

        # Thin separator
        ctk.CTkFrame(self._sidebar, height=1, fg_color=BORDER).grid(
            row=1, column=0, sticky="ew", padx=16, pady=(0, 8)
        )

        # Nav buttons
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for btn_row, (key, label, icon) in enumerate(_NAV_ITEMS, start=2):
            btn = ctk.CTkButton(
                self._sidebar,
                text=f"{icon}   {label}",
                anchor="w",
                font=("Segoe UI", 13),
                height=42,
                fg_color="transparent",
                text_color=SUBTEXT,
                hover_color=CARD,
                corner_radius=8,
                command=lambda k=key: self.show_page(k),
            )
            btn.grid(row=btn_row, column=0, padx=10, pady=2, sticky="ew")
            self._nav_buttons[key] = btn

        # Version label at bottom
        ctk.CTkLabel(
            self._sidebar, text="v1.0.0",
            font=("Segoe UI", 10), text_color=SUBTEXT,
        ).grid(row=9, column=0, pady=(0, 14))

        # ── Content area ─────────────────────────────────────────────────
        content_area = ctk.CTkFrame(self, fg_color=CONTENT, corner_radius=0)
        content_area.grid(row=0, column=1, sticky="nsew")
        content_area.columnconfigure(0, weight=1)
        content_area.rowconfigure(0, weight=1)

        # ── Instantiate all pages ────────────────────────────────────────
        self._pages: dict[str, ctk.CTkFrame] = {}

        self._pages["dashboard"] = DashboardPage(
            content_area, app=self, history_service=history_service
        )
        self._pages["predict"] = PredictPage(
            content_area, app=self,
            prediction_service=prediction_service,
            image_service=image_service,
            history_service=history_service,
        )
        self._pages["history"] = HistoryPage(
            content_area, app=self, history_service=history_service
        )
        self._pages["dataset"] = DatasetPage(
            content_area, app=self
        )
        self._pages["model"] = ModelPage(
            content_area, app=self, model_service=model_service
        )
        self._pages["settings"] = SettingsPage(
            content_area, app=self
        )
        self._pages["about"] = AboutPage(
            content_area, app=self
        )

        # Place all pages in the same grid cell; only the active one is visible
        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_remove()

        self._current_page: str | None = None

        # ── Start on dashboard ───────────────────────────────────────────
        self.show_page("dashboard")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_page(self, name: str) -> None:
        """Hide current page, show requested page, call page.refresh()."""
        # Hide current
        if self._current_page and self._current_page in self._pages:
            self._pages[self._current_page].grid_remove()

        # Show new
        if name not in self._pages:
            return
        self._pages[name].grid()
        self._current_page = name

        # Update nav button styles
        self._set_active_nav(name)

        # Refresh page data
        try:
            self._pages[name].refresh()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Page '%s' refresh() raised: %s", name, exc
            )

    def _set_active_nav(self, active_key: str) -> None:
        for key, btn in self._nav_buttons.items():
            if key == active_key:
                btn.configure(fg_color=CARD, text_color=ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color=SUBTEXT)
