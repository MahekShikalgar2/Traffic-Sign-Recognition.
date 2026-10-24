"""
DashboardPage — overview cards, quick actions, and recent predictions.
"""

from datetime import datetime
import customtkinter as ctk

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER,
    CONTENT, SIDEBAR,
)


class DashboardPage(ctk.CTkFrame):
    """Main dashboard page with stats, quick actions, and recent predictions."""

    def __init__(self, master, app, history_service, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._history_service = history_service
        self._stat_labels = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(28, 6))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Traffic Sign Recognition",
            font=("Segoe UI", 26, "bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="AI-Powered Computer Vision Dashboard",
            font=("Segoe UI", 13),
            text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Stat cards row ───────────────────────────────────────────────
        cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="ew", padx=30, pady=(18, 0))
        for i in range(4):
            cards_frame.columnconfigure(i, weight=1)

        self._stat_labels["total_predictions"] = self._make_stat_card(
            cards_frame, col=0,
            icon="🔍", title="Total Predictions", value="0", color=ACCENT,
        )
        self._make_stat_card(
            cards_frame, col=1,
            icon="🖼", title="Dataset Images", value="34,776", color=ACCENT2,
        )
        self._make_stat_card(
            cards_frame, col=2,
            icon="🏷", title="Sign Classes", value="43", color=SUCCESS,
        )
        self._make_stat_card(
            cards_frame, col=3,
            icon="📈", title="Training Accuracy", value="~96%", color=WARNING,
        )

        # ── Quick actions + recent predictions ──────────────────────────
        lower = ctk.CTkFrame(self, fg_color="transparent")
        lower.grid(row=2, column=0, sticky="nsew", padx=30, pady=18)
        lower.columnconfigure(0, weight=0)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        self._build_quick_actions(lower)
        self._build_recent_panel(lower)

    def _make_stat_card(self, parent, col, icon, title, value, color):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12)
        card.grid(row=0, column=col, sticky="ew", padx=(0, 12) if col < 3 else 0, pady=4)
        card.columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text=icon, font=("Segoe UI", 24)).grid(
            row=0, column=0, padx=20, pady=(16, 4), sticky="w"
        )
        value_lbl = ctk.CTkLabel(
            card, text=value, font=("Segoe UI", 22, "bold"), text_color=color
        )
        value_lbl.grid(row=1, column=0, padx=20, sticky="w")
        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 11), text_color=SUBTEXT
        ).grid(row=2, column=0, padx=20, pady=(2, 16), sticky="w")

        return value_lbl   # return so we can update it later

    def _build_quick_actions(self, parent) -> None:
        frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, width=220)
        frame.grid(row=0, column=0, sticky="ns", padx=(0, 16))
        frame.columnconfigure(0, weight=1)
        frame.grid_propagate(False)

        ctk.CTkLabel(
            frame, text="Quick Actions", font=("Segoe UI", 14, "bold"), text_color=TEXT
        ).grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkButton(
            frame,
            text="📷  Upload & Predict",
            font=("Segoe UI", 13),
            fg_color=ACCENT,
            hover_color="#0EA5E9",
            text_color="#0B1120",
            corner_radius=8,
            height=40,
            command=lambda: self._app.show_page("predict"),
        ).grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")

        ctk.CTkButton(
            frame,
            text="📋  View History",
            font=("Segoe UI", 13),
            fg_color=ACCENT2,
            hover_color="#7C3AED",
            text_color=TEXT,
            corner_radius=8,
            height=40,
            command=lambda: self._app.show_page("history"),
        ).grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        ctk.CTkButton(
            frame,
            text="📊  Dataset Info",
            font=("Segoe UI", 13),
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            hover_color="#2D3748",
            text_color=SUBTEXT,
            corner_radius=8,
            height=40,
            command=lambda: self._app.show_page("dataset"),
        ).grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")

        ctk.CTkButton(
            frame,
            text="🧠  Model Info",
            font=("Segoe UI", 13),
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            hover_color="#2D3748",
            text_color=SUBTEXT,
            corner_radius=8,
            height=40,
            command=lambda: self._app.show_page("model"),
        ).grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")

    def _build_recent_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Header row
        hdr = ctk.CTkFrame(frame, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr, text="Recent Predictions", font=("Segoe UI", 14, "bold"), text_color=TEXT
        ).grid(row=0, column=0, sticky="w")

        self._recent_scroll = ctk.CTkScrollableFrame(
            frame, fg_color="transparent", scrollbar_button_color=BORDER
        )
        self._recent_scroll.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 8))
        self._recent_scroll.columnconfigure(0, weight=1)

        self._no_history_lbl = ctk.CTkLabel(
            self._recent_scroll,
            text="No predictions yet.\nUpload an image on the Predict page to get started.",
            font=("Segoe UI", 13),
            text_color=SUBTEXT,
            justify="center",
        )
        self._no_history_lbl.grid(row=0, column=0, padx=20, pady=40)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _clear_recent(self) -> None:
        for widget in self._recent_scroll.winfo_children():
            widget.destroy()

    def _populate_recent(self, records: list) -> None:
        self._clear_recent()
        if not records:
            ctk.CTkLabel(
                self._recent_scroll,
                text="No predictions yet.\nUpload an image on the Predict page to get started.",
                font=("Segoe UI", 13),
                text_color=SUBTEXT,
                justify="center",
            ).grid(row=0, column=0, padx=20, pady=40)
            return

        for i, rec in enumerate(records[:5]):
            self._add_recent_row(i, rec)

    def _add_recent_row(self, row_idx: int, rec: dict) -> None:
        bg = CARD if row_idx % 2 == 0 else "#243047"
        row = ctk.CTkFrame(self._recent_scroll, fg_color=bg, corner_radius=8)
        row.grid(row=row_idx, column=0, sticky="ew", padx=8, pady=3)
        row.columnconfigure(1, weight=1)

        # Index badge
        ctk.CTkLabel(
            row, text=f"#{row_idx + 1}", font=("Segoe UI", 11),
            text_color=SUBTEXT, width=30
        ).grid(row=0, column=0, padx=(12, 8), pady=10)

        # Sign name
        ctk.CTkLabel(
            row, text=rec.get("predicted_label", "Unknown"),
            font=("Segoe UI", 12, "bold"), text_color=TEXT, anchor="w"
        ).grid(row=0, column=1, sticky="w", pady=10)

        # Confidence badge
        conf = rec.get("confidence", 0.0)
        conf_color = SUCCESS if conf >= 0.8 else (WARNING if conf >= 0.6 else ERROR)
        ctk.CTkLabel(
            row, text=f"{conf * 100:.1f}%",
            font=("Segoe UI", 11, "bold"), text_color=conf_color
        ).grid(row=0, column=2, padx=(8, 8), pady=10)

        # Timestamp
        ts = rec.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%m/%d %H:%M")
        except Exception:
            ts_str = ts[:16] if len(ts) >= 16 else ts
        ctk.CTkLabel(
            row, text=ts_str, font=("Segoe UI", 10), text_color=SUBTEXT
        ).grid(row=0, column=3, padx=(0, 12), pady=10)

    # ------------------------------------------------------------------
    # Refresh (called by App.show_page)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload stats and recent predictions from HistoryService."""
        try:
            stats = self._history_service.get_stats()
            total = stats.get("total_predictions", 0)
            self._stat_labels["total_predictions"].configure(text=str(total))
        except Exception:
            pass

        try:
            records = self._history_service.get_all()
            self._populate_recent(records)
        except Exception:
            pass
