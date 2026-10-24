"""
HistoryPage — lists all past predictions with search, export, and clear.
"""

from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER, CONTENT,
)


class HistoryPage(ctk.CTkFrame):
    """Prediction history browser with search, export, and detail view."""

    def __init__(self, master, app, history_service, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._history_service = history_service
        self._records: list = []
        self._selected_row: ctk.CTkFrame | None = None
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
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Prediction History",
            font=("Segoe UI", 24, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="All past predictions sorted by most recent",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Action buttons (top-right)
        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.grid(row=0, column=1, rowspan=2, sticky="e")

        ctk.CTkButton(
            actions, text="📥  Export CSV",
            font=("Segoe UI", 12), height=34, corner_radius=7,
            fg_color=ACCENT2, hover_color="#7C3AED", text_color=TEXT,
            command=self._on_export,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            actions, text="🗑  Clear History",
            font=("Segoe UI", 12), height=34, corner_radius=7,
            fg_color=CARD, border_width=1, border_color=ERROR,
            hover_color="#450A0A", text_color=ERROR,
            command=self._on_clear,
        ).pack(side="left")

        # ── Search bar ───────────────────────────────────────────────────
        search_bar = ctk.CTkFrame(self, fg_color="transparent")
        search_bar.grid(row=1, column=0, sticky="ew", padx=30, pady=(14, 0))
        search_bar.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            search_bar, text="🔎", font=("Segoe UI", 14), text_color=SUBTEXT,
        ).grid(row=0, column=0, padx=(0, 8))

        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._apply_filter())
        ctk.CTkEntry(
            search_bar,
            textvariable=self._search_var,
            placeholder_text="Search by sign name or class ID…",
            font=("Segoe UI", 12),
            fg_color=CARD, border_color=BORDER,
            height=36, corner_radius=8,
        ).grid(row=0, column=1, sticky="ew")

        # ── Main area (list + detail) ────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=2, column=0, sticky="nsew", padx=30, pady=14)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        # List panel
        list_outer = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12)
        list_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        list_outer.columnconfigure(0, weight=1)
        list_outer.rowconfigure(1, weight=1)

        # Column headers
        col_hdr = ctk.CTkFrame(list_outer, fg_color="#243047", corner_radius=0)
        col_hdr.grid(row=0, column=0, sticky="ew")
        for col_i, (text, w) in enumerate([
            ("#", 35), ("Sign Name", 0), ("ID", 50), ("Conf.", 60), ("Date/Time", 130)
        ]):
            ctk.CTkLabel(
                col_hdr, text=text,
                font=("Segoe UI", 11, "bold"), text_color=SUBTEXT,
                width=w if w else 0, anchor="w",
            ).grid(row=0, column=col_i, padx=(12 if col_i == 0 else 4, 4),
                   pady=8, sticky="w")
            if w == 0:
                col_hdr.columnconfigure(col_i, weight=1)

        self._list_scroll = ctk.CTkScrollableFrame(
            list_outer, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        self._list_scroll.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self._list_scroll.columnconfigure(0, weight=1)

        # Detail panel (fixed width)
        self._detail_frame = ctk.CTkFrame(main, fg_color=CARD, corner_radius=12, width=280)
        self._detail_frame.grid(row=0, column=1, sticky="ns")
        self._detail_frame.grid_propagate(False)
        self._detail_frame.columnconfigure(0, weight=1)
        self._show_detail_placeholder()

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------

    def _clear_list(self) -> None:
        for w in self._list_scroll.winfo_children():
            w.destroy()

    def _apply_filter(self) -> None:
        query = self._search_var.get().lower().strip()
        if not query:
            filtered = self._records
        else:
            filtered = [
                r for r in self._records
                if query in r.get("predicted_label", "").lower()
                or query in str(r.get("predicted_class", ""))
            ]
        self._render_list(filtered)

    def _render_list(self, records: list) -> None:
        self._clear_list()
        if not records:
            ctk.CTkLabel(
                self._list_scroll,
                text="No predictions found." if self._records else
                     "No predictions yet.\nUpload an image to get started.",
                font=("Segoe UI", 13), text_color=SUBTEXT, justify="center",
            ).grid(row=0, column=0, padx=20, pady=50)
            return

        for i, rec in enumerate(records):
            self._add_list_row(i, rec)

    def _add_list_row(self, row_idx: int, rec: dict) -> None:
        bg = CARD if row_idx % 2 == 0 else "#1A2744"
        row = ctk.CTkFrame(self._list_scroll, fg_color=bg, corner_radius=6)
        row.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=2)
        row.columnconfigure(1, weight=1)

        conf = rec.get("confidence", 0.0)
        conf_color = SUCCESS if conf >= 0.8 else (WARNING if conf >= 0.6 else ERROR)

        ts = rec.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%m/%d/%y %H:%M")
        except Exception:
            ts_str = ts[:16]

        cells = [
            (f"{row_idx + 1}", 35, SUBTEXT, "center"),
            (rec.get("predicted_label", "Unknown"), 0, TEXT, "w"),
            (str(rec.get("predicted_class", "?")), 50, SUBTEXT, "center"),
            (f"{conf * 100:.0f}%", 60, conf_color, "center"),
            (ts_str, 130, SUBTEXT, "w"),
        ]
        for col_i, (text, width, color, anchor) in enumerate(cells):
            kw = {"width": width} if width else {}
            ctk.CTkLabel(
                row, text=text, font=("Segoe UI", 11),
                text_color=color, anchor=anchor, **kw,
            ).grid(
                row=0, column=col_i,
                padx=(12 if col_i == 0 else 4, 4), pady=8,
                sticky="w" if not width else "",
            )
            if not width:
                row.columnconfigure(col_i, weight=1)

        # Click to show detail
        row.bind("<Button-1>", lambda e, r=rec, f=row: self._on_row_click(r, f))
        for child in row.winfo_children():
            child.bind("<Button-1>", lambda e, r=rec, f=row: self._on_row_click(r, f))

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------

    def _show_detail_placeholder(self) -> None:
        for w in self._detail_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._detail_frame, text="Select a row\nto see details",
            font=("Segoe UI", 12), text_color=SUBTEXT, justify="center",
        ).place(relx=0.5, rely=0.5, anchor="center")

    def _on_row_click(self, rec: dict, frame: ctk.CTkFrame) -> None:
        # Highlight selected row (reset previous)
        if self._selected_row:
            try:
                self._selected_row.configure(border_width=0)
            except Exception:
                pass
        self._selected_row = frame
        frame.configure(border_width=1, border_color=ACCENT)
        self._show_detail(rec)

    def _show_detail(self, rec: dict) -> None:
        for w in self._detail_frame.winfo_children():
            w.destroy()

        scroll = ctk.CTkScrollableFrame(
            self._detail_frame, fg_color="transparent",
            scrollbar_button_color=BORDER,
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._detail_frame.rowconfigure(0, weight=1)
        scroll.columnconfigure(0, weight=1)

        row_i = 0

        ctk.CTkLabel(
            scroll, text="Prediction Detail",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=row_i, column=0, padx=14, pady=(14, 8), sticky="w")
        row_i += 1

        conf = rec.get("confidence", 0.0)
        conf_color = SUCCESS if conf >= 0.8 else (WARNING if conf >= 0.6 else ERROR)

        fields = [
            ("Sign", rec.get("predicted_label", "—")),
            ("Class ID", str(rec.get("predicted_class", "—"))),
            ("Confidence", f"{conf * 100:.2f}%"),
            ("File", rec.get("filename", "—")),
            ("Time", rec.get("timestamp", "—")[:19]),
            ("Proc. Time", f"{rec.get('processing_time_ms', 0):.0f} ms"),
        ]

        for label, value in fields:
            f = ctk.CTkFrame(scroll, fg_color="transparent")
            f.grid(row=row_i, column=0, sticky="ew", padx=10, pady=3)
            f.columnconfigure(1, weight=1)
            ctk.CTkLabel(
                f, text=label + ":", width=70,
                font=("Segoe UI", 10), text_color=SUBTEXT, anchor="w",
            ).grid(row=0, column=0, sticky="w")
            color = conf_color if label == "Confidence" else TEXT
            ctk.CTkLabel(
                f, text=value,
                font=("Segoe UI", 10, "bold"), text_color=color,
                anchor="w", wraplength=150,
            ).grid(row=0, column=1, sticky="w")
            row_i += 1

        # Top-3
        top3 = rec.get("top3", [])
        if top3:
            ctk.CTkLabel(
                scroll, text="Top 3",
                font=("Segoe UI", 11, "bold"), text_color=TEXT,
            ).grid(row=row_i, column=0, padx=14, pady=(10, 4), sticky="w")
            row_i += 1
            for j, entry in enumerate(top3):
                lbl = entry.get("label", "Unknown")
                prob = entry.get("probability", 0.0)
                t_row = ctk.CTkFrame(scroll, fg_color="transparent")
                t_row.grid(row=row_i, column=0, sticky="ew", padx=10, pady=2)
                t_row.columnconfigure(1, weight=1)
                ctk.CTkLabel(
                    t_row, text=f"{j + 1}.", width=18,
                    font=("Segoe UI", 10), text_color=SUBTEXT,
                ).grid(row=0, column=0)
                ctk.CTkLabel(
                    t_row, text=lbl, anchor="w",
                    font=("Segoe UI", 10), text_color=TEXT,
                ).grid(row=0, column=1, sticky="w", padx=(4, 0))
                ctk.CTkLabel(
                    t_row, text=f"{prob * 100:.1f}%",
                    font=("Segoe UI", 10, "bold"), text_color=ACCENT,
                ).grid(row=0, column=2, padx=(4, 0))
                row_i += 1

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export History CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="prediction_history.csv",
        )
        if path:
            try:
                self._history_service.export_csv(path)
                messagebox.showinfo("Export Successful", f"History exported to:\n{path}", parent=self)
            except Exception as exc:
                messagebox.showerror("Export Failed", str(exc), parent=self)

    def _on_clear(self) -> None:
        if not self._records:
            messagebox.showinfo("Nothing to Clear", "History is already empty.", parent=self)
            return
        answer = messagebox.askyesno(
            "Clear History",
            "Are you sure you want to delete all prediction history?\n\nThis cannot be undone.",
            icon="warning",
            parent=self,
        )
        if answer:
            self._history_service.clear_history()
            self.refresh()

    # ------------------------------------------------------------------
    # Refresh (called by App.show_page)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._records = self._history_service.get_all()
        self._apply_filter()
        self._show_detail_placeholder()
