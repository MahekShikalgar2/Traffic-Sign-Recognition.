"""
DatasetPage — dataset statistics and class distribution chart.

matplotlib backend must be set to TkAgg BEFORE pyplot is imported.
"""

import sys

import matplotlib
matplotlib.use("TkAgg")   # must be set before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER, CONTENT,
    DATASET_CLASS_COUNTS, NUM_CLASSES,
)

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import signs as _signs


class DatasetPage(ctk.CTkFrame):
    """Dataset statistics with embedded matplotlib bar chart."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._chart_drawn = False
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(28, 0))
        ctk.CTkLabel(
            header, text="Dataset Information",
            font=("Segoe UI", 24, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="GTSRB — German Traffic Sign Recognition Benchmark",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Stats cards ─────────────────────────────────────────────────
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", padx=30, pady=(18, 0))
        for i in range(4):
            cards.columnconfigure(i, weight=1)

        stats = [
            ("🖼", "Total Images", "34,776", ACCENT),
            ("🏷", "Classes", "43", ACCENT2),
            ("📂", "Source", "GTSRB", SUCCESS),
            ("📐", "Input Size", "32 × 32 px", WARNING),
        ]
        for col, (icon, title, val, color) in enumerate(stats):
            card = ctk.CTkFrame(cards, fg_color=CARD, corner_radius=10)
            card.grid(row=0, column=col, sticky="ew", padx=(0, 10) if col < 3 else 0, pady=4)
            ctk.CTkLabel(card, text=icon, font=("Segoe UI", 22)).grid(
                row=0, column=0, padx=16, pady=(14, 4), sticky="w"
            )
            ctk.CTkLabel(
                card, text=val, font=("Segoe UI", 20, "bold"), text_color=color
            ).grid(row=1, column=0, padx=16, sticky="w")
            ctk.CTkLabel(
                card, text=title, font=("Segoe UI", 11), text_color=SUBTEXT
            ).grid(row=2, column=0, padx=16, pady=(2, 14), sticky="w")

        # ── Lower area: chart + class browser ───────────────────────────
        lower = ctk.CTkFrame(self, fg_color="transparent")
        lower.grid(row=2, column=0, sticky="nsew", padx=30, pady=14)
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=0)
        lower.rowconfigure(0, weight=1)

        # Chart card
        chart_card = ctk.CTkFrame(lower, fg_color=CARD, corner_radius=12)
        chart_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        chart_card.columnconfigure(0, weight=1)
        chart_card.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            chart_card, text="Class Distribution",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, padx=20, pady=(16, 8), sticky="w")

        self._chart_frame = ctk.CTkFrame(chart_card, fg_color=CARD, corner_radius=0)
        self._chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 12))
        self._chart_frame.columnconfigure(0, weight=1)
        self._chart_frame.rowconfigure(0, weight=1)

        # Class browser
        browser_card = ctk.CTkFrame(lower, fg_color=CARD, corner_radius=12, width=250)
        browser_card.grid(row=0, column=1, sticky="ns")
        browser_card.grid_propagate(False)
        browser_card.columnconfigure(0, weight=1)
        browser_card.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            browser_card, text="All Classes (43)",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        browser_scroll = ctk.CTkScrollableFrame(
            browser_card, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        browser_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 12))
        browser_scroll.columnconfigure(0, weight=1)

        for class_id in range(NUM_CLASSES):
            sign_name = _signs.classes.get(class_id + 1, f"Class {class_id}")
            count = DATASET_CLASS_COUNTS.get(class_id, 0)
            row_frame = ctk.CTkFrame(
                browser_scroll,
                fg_color=CARD if class_id % 2 == 0 else "#1A2744",
                corner_radius=6,
            )
            row_frame.grid(row=class_id, column=0, sticky="ew", padx=4, pady=2)
            row_frame.columnconfigure(1, weight=1)

            ctk.CTkLabel(
                row_frame, text=str(class_id), width=28,
                font=("Segoe UI", 10), text_color=ACCENT, anchor="center",
            ).grid(row=0, column=0, padx=(10, 4), pady=6)

            ctk.CTkLabel(
                row_frame, text=sign_name, anchor="w",
                font=("Segoe UI", 10), text_color=TEXT,
            ).grid(row=0, column=1, sticky="w")

            ctk.CTkLabel(
                row_frame, text=str(count), width=40,
                font=("Segoe UI", 10), text_color=SUBTEXT, anchor="e",
            ).grid(row=0, column=2, padx=(0, 10))

    def _draw_chart(self) -> None:
        """Draw the bar chart once and embed it in the chart frame."""
        if self._chart_drawn:
            return
        self._chart_drawn = True

        ids = list(DATASET_CLASS_COUNTS.keys())
        counts = [DATASET_CLASS_COUNTS[k] for k in ids]

        bg_hex = CARD   # #1E293B
        # Convert hex to 0–1 RGB for matplotlib
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(bg_hex)
        text_rgb = hex_to_rgb(TEXT)
        accent_rgb = hex_to_rgb(ACCENT)
        subtext_rgb = hex_to_rgb(SUBTEXT)

        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=90)
        fig.patch.set_facecolor(bg_rgb)
        ax.set_facecolor(bg_rgb)

        bars = ax.bar(ids, counts, color=accent_rgb, width=0.7, zorder=3)

        # Accent a few bars differently
        for bar, count in zip(bars, counts):
            if count == max(counts):
                bar.set_facecolor(hex_to_rgb(ACCENT2))
            elif count == min(counts):
                bar.set_facecolor(hex_to_rgb(ERROR))

        ax.set_xlabel("Class ID", color=subtext_rgb, fontsize=9)
        ax.set_ylabel("Images", color=subtext_rgb, fontsize=9)
        ax.tick_params(colors=subtext_rgb, labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(BORDER)
        ax.grid(axis="y", color=BORDER, linewidth=0.5, zorder=0)
        ax.set_xlim(-1, NUM_CLASSES)
        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._draw_chart()
