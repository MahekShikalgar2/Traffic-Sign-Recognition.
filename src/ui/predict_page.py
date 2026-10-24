"""
PredictPage — image upload, preprocessing, CNN inference, and result display.

Thread safety:
  model.predict() runs in a background threading.Thread.
  All CTk widget updates are dispatched via app.after(0, callback).
"""

import threading
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER,
    CONTENT, SUPPORTED_FORMATS, CONFIDENCE_THRESHOLD,
)
from src.utils.validators import ImageValidator

try:
    from src.sign_info import SIGN_INFO
except ImportError:
    SIGN_INFO = {}

try:
    import tkinterdnd2  # type: ignore
    HAVE_DND = True
except ImportError:
    HAVE_DND = False


class PredictPage(ctk.CTkFrame):
    """Main prediction page: upload + preview on left, results on right."""

    def __init__(self, master, app, prediction_service, image_service,
                 history_service, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._prediction_service = prediction_service
        self._image_service = image_service
        self._history_service = history_service

        self._pil_image: Image.Image | None = None
        self._image_path: str | None = None
        self._is_predicting: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)   # header
        self.rowconfigure(1, weight=1)   # content

        # ── Page header ─────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=30, pady=(28, 0))
        ctk.CTkLabel(
            header, text="Predict Traffic Sign",
            font=("Segoe UI", 24, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Upload an image and run CNN inference",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Left panel (upload) ─────────────────────────────────────────
        left = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12, width=460)
        left.grid(row=1, column=0, sticky="nsew", padx=(30, 10), pady=20)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)
        left.grid_propagate(False)
        self._build_left_panel(left)

        # ── Right panel (results) ───────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 30), pady=20)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self._right_frame = right
        self._show_placeholder()

    def _build_left_panel(self, frame: ctk.CTkFrame) -> None:
        """Upload zone, file info, preview, and predict button."""

        # ── Title ────────────────────────────────────────────────────────
        ctk.CTkLabel(
            frame, text="Image Upload",
            font=("Segoe UI", 14, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, padx=20, pady=(20, 0), sticky="w")

        # ── Drop zone ────────────────────────────────────────────────────
        self._drop_zone = ctk.CTkFrame(
            frame, fg_color="#172033", corner_radius=10,
            border_width=2, border_color=BORDER,
        )
        self._drop_zone.grid(row=1, column=0, padx=20, pady=12, sticky="ew")
        self._drop_zone.columnconfigure(0, weight=1)

        self._drop_icon_lbl = ctk.CTkLabel(
            self._drop_zone, text="📁", font=("Segoe UI", 32)
        )
        self._drop_icon_lbl.grid(row=0, column=0, pady=(18, 4))

        self._drop_text_lbl = ctk.CTkLabel(
            self._drop_zone,
            text="Drop image here or click Upload\nSupported: JPG, JPEG, PNG, BMP",
            font=("Segoe UI", 12), text_color=SUBTEXT, justify="center",
        )
        self._drop_text_lbl.grid(row=1, column=0, pady=(0, 14))

        # Register DnD if available
        if HAVE_DND:
            try:
                self._drop_zone.drop_target_register("DND_Files")  # type: ignore
                self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
            except Exception:
                pass

        # ── Upload / Clear buttons ────────────────────────────────────────
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=20, pady=(0, 12), sticky="ew")
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._upload_btn = ctk.CTkButton(
            btn_row, text="📂  Upload Image",
            font=("Segoe UI", 13),
            fg_color=ACCENT, hover_color="#0EA5E9", text_color="#0B1120",
            corner_radius=8, height=38,
            command=self._on_upload,
        )
        self._upload_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self._clear_btn = ctk.CTkButton(
            btn_row, text="✕  Clear",
            font=("Segoe UI", 13),
            fg_color=CARD, hover_color="#374151",
            border_width=1, border_color=BORDER,
            text_color=SUBTEXT, corner_radius=8, height=38,
            state="disabled",
            command=self._on_clear,
        )
        self._clear_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # ── File info section ────────────────────────────────────────────
        self._info_frame = ctk.CTkFrame(frame, fg_color="#172033", corner_radius=8)
        self._info_frame.grid(row=3, column=0, padx=20, pady=(0, 12), sticky="ew")
        self._info_frame.columnconfigure(1, weight=1)
        # hidden until image loaded
        self._info_frame.grid_remove()

        info_fields = [
            ("📄  File:", "filename"),
            ("📐  Size:", "dimensions"),
            ("💾  File size:", "file_size"),
            ("🎨  Format:", "format"),
        ]
        self._info_vals = {}
        for i, (label, key) in enumerate(info_fields):
            ctk.CTkLabel(
                self._info_frame, text=label,
                font=("Segoe UI", 11), text_color=SUBTEXT,
            ).grid(row=i, column=0, padx=(14, 6), pady=4, sticky="w")
            lbl = ctk.CTkLabel(
                self._info_frame, text="—",
                font=("Segoe UI", 11), text_color=TEXT,
            )
            lbl.grid(row=i, column=1, padx=(0, 14), pady=4, sticky="w")
            self._info_vals[key] = lbl

        # ── Image preview ────────────────────────────────────────────────
        self._preview_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self._preview_frame.grid(row=4, column=0, padx=20, pady=(0, 12), sticky="nsew")
        self._preview_frame.columnconfigure(0, weight=1)

        self._preview_lbl = ctk.CTkLabel(
            self._preview_frame, text="", image=None,
        )
        self._preview_lbl.grid(row=0, column=0)

        # ── Predict button ────────────────────────────────────────────────
        self._predict_btn = ctk.CTkButton(
            frame, text="🔍  Predict Sign",
            font=("Segoe UI", 14, "bold"),
            fg_color=ACCENT, hover_color="#0EA5E9", text_color="#0B1120",
            corner_radius=8, height=44,
            state="disabled",
            command=self._on_predict,
        )
        self._predict_btn.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="ew")

        # ── Busy progress bar ─────────────────────────────────────────────
        self._progress = ctk.CTkProgressBar(
            frame, mode="indeterminate",
            fg_color=BORDER, progress_color=ACCENT,
            height=4,
        )
        self._progress.grid(row=6, column=0, padx=20, pady=(0, 16), sticky="ew")
        self._progress.grid_remove()

    # ------------------------------------------------------------------
    # Placeholder / result display in right panel
    # ------------------------------------------------------------------

    def _clear_right(self) -> None:
        for w in self._right_frame.winfo_children():
            w.destroy()

    def _show_placeholder(self) -> None:
        self._clear_right()
        ph = ctk.CTkFrame(self._right_frame, fg_color="transparent")
        ph.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(ph, text="🔍", font=("Segoe UI", 48)).pack()
        ctk.CTkLabel(
            ph, text="No prediction yet",
            font=("Segoe UI", 16, "bold"), text_color=TEXT,
        ).pack(pady=(8, 4))
        ctk.CTkLabel(
            ph, text="Upload an image and click Predict Sign\nto see the result here.",
            font=("Segoe UI", 13), text_color=SUBTEXT, justify="center",
        ).pack()

    def _show_result(self, result: dict, filename: str) -> None:
        """Render prediction result card in the right panel."""
        self._clear_right()

        scroll = ctk.CTkScrollableFrame(
            self._right_frame, fg_color="transparent",
            scrollbar_button_color=BORDER,
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scroll.columnconfigure(0, weight=1)

        conf = result.get("confidence", 0.0)
        label = result.get("predicted_label", "Unknown")
        class_id = result.get("predicted_class", -1)
        top3 = result.get("top3", [])
        ts = result.get("timestamp", datetime.now().isoformat())
        proc_ms = result.get("processing_time_ms", 0.0)
        is_low = result.get("is_low_confidence", False)

        # Colour indicator
        if conf >= 0.80:
            conf_color = SUCCESS
            badge_bg = "#14532D"
            status_text = "High Confidence"
        elif conf >= 0.60:
            conf_color = WARNING
            badge_bg = "#451A03"
            status_text = "Moderate Confidence"
        else:
            conf_color = ERROR
            badge_bg = "#450A0A"
            status_text = "Low Confidence"

        row_idx = 0

        # ── Result header ─────────────────────────────────────────────
        hdr = ctk.CTkFrame(scroll, fg_color="transparent")
        hdr.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(20, 6))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr, text="Prediction Result",
            font=("Segoe UI", 14, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        row_idx += 1

        # ── Main sign card ────────────────────────────────────────────
        main_card = ctk.CTkFrame(scroll, fg_color="#1A2744", corner_radius=10,
                                  border_width=1, border_color=conf_color)
        main_card.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=6)
        main_card.columnconfigure(0, weight=1)
        row_idx += 1

        ctk.CTkLabel(
            main_card, text="🚦", font=("Segoe UI", 36),
        ).grid(row=0, column=0, padx=20, pady=(16, 4), sticky="w")

        ctk.CTkLabel(
            main_card, text=label,
            font=("Segoe UI", 18, "bold"), text_color=TEXT, anchor="w",
        ).grid(row=1, column=0, padx=20, sticky="w")

        # Class ID badge + confidence
        badges = ctk.CTkFrame(main_card, fg_color="transparent")
        badges.grid(row=2, column=0, padx=20, pady=(6, 0), sticky="w")

        class_badge = ctk.CTkFrame(badges, fg_color=CARD, corner_radius=6)
        class_badge.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            class_badge, text=f"Class {class_id}",
            font=("Segoe UI", 11), text_color=SUBTEXT,
        ).pack(padx=10, pady=4)

        status_badge = ctk.CTkFrame(badges, fg_color=badge_bg, corner_radius=6)
        status_badge.pack(side="left")
        ctk.CTkLabel(
            status_badge, text=status_text,
            font=("Segoe UI", 11, "bold"), text_color=conf_color,
        ).pack(padx=10, pady=4)

        # Confidence percentage + bar
        ctk.CTkLabel(
            main_card, text=f"{conf * 100:.1f}% confidence",
            font=("Segoe UI", 22, "bold"), text_color=conf_color,
        ).grid(row=3, column=0, padx=20, pady=(10, 4), sticky="w")

        conf_bar = ctk.CTkProgressBar(
            main_card, fg_color=BORDER, progress_color=conf_color, height=8, corner_radius=4
        )
        conf_bar.grid(row=4, column=0, padx=20, pady=(0, 4), sticky="ew")
        conf_bar.set(conf)

        # Low-confidence warning
        if is_low:
            warn_frame = ctk.CTkFrame(main_card, fg_color="#451A03", corner_radius=6)
            warn_frame.grid(row=5, column=0, padx=20, pady=(8, 0), sticky="ew")
            ctk.CTkLabel(
                warn_frame,
                text="⚠  Low confidence — please upload a clearer image.",
                font=("Segoe UI", 11), text_color=WARNING,
            ).grid(padx=12, pady=8, sticky="w")
            row_num = 6
        else:
            row_num = 5

        # Timestamp / time
        meta = ctk.CTkFrame(main_card, fg_color="transparent")
        meta.grid(row=row_num, column=0, padx=20, pady=(8, 16), sticky="w")
        try:
            dt_obj = datetime.fromisoformat(ts)
            ts_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = ts[:19]
        ctk.CTkLabel(
            meta, text=f"🕒 {ts_str}   ⚡ {proc_ms:.0f} ms",
            font=("Segoe UI", 10), text_color=SUBTEXT,
        ).pack(side="left")

        # ── Top-3 predictions ─────────────────────────────────────────
        row_idx += 1
        ctk.CTkLabel(
            scroll, text="Top 3 Predictions",
            font=("Segoe UI", 13, "bold"), text_color=TEXT,
        ).grid(row=row_idx, column=0, padx=20, pady=(14, 6), sticky="w")
        row_idx += 1

        top3_card = ctk.CTkFrame(scroll, fg_color="#1A2744", corner_radius=10)
        top3_card.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(0, 6))
        top3_card.columnconfigure(0, weight=1)
        row_idx += 1

        bar_colors = [ACCENT, ACCENT2, "#64748B"]
        for i, entry in enumerate(top3):
            prob = entry.get("probability", 0.0)
            lbl_text = entry.get("label", "Unknown")
            cid = entry.get("class_id", -1)
            self._make_top3_row(top3_card, i, i + 1, lbl_text, cid, prob, bar_colors[i])

        # ── Sign Information card ─────────────────────────────────────
        sign_data = SIGN_INFO.get(class_id)
        if sign_data:
            ctk.CTkLabel(
                scroll, text="Sign Information",
                font=("Segoe UI", 13, "bold"), text_color=TEXT,
            ).grid(row=row_idx, column=0, padx=20, pady=(14, 6), sticky="w")
            row_idx += 1

            info_card = ctk.CTkFrame(scroll, fg_color="#1A2744", corner_radius=10,
                                     border_width=1, border_color=BORDER)
            info_card.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(0, 6))
            info_card.columnconfigure(1, weight=1)
            row_idx += 1

            # Category badge
            cat_row = ctk.CTkFrame(info_card, fg_color="transparent")
            cat_row.grid(row=0, column=0, columnspan=2, padx=14, pady=(14, 4), sticky="w")
            cat_badge = ctk.CTkFrame(cat_row, fg_color=CARD, corner_radius=6)
            cat_badge.pack(side="left")
            ctk.CTkLabel(
                cat_badge, text=f"🏷  {sign_data['category']}",
                font=("Segoe UI", 11), text_color=ACCENT,
            ).pack(padx=10, pady=4)

            # Meaning row
            ctk.CTkLabel(
                info_card, text="Meaning:",
                font=("Segoe UI", 11, "bold"), text_color=SUBTEXT, anchor="nw",
            ).grid(row=1, column=0, padx=(14, 8), pady=(8, 2), sticky="nw")
            ctk.CTkLabel(
                info_card, text=sign_data["meaning"],
                font=("Segoe UI", 11), text_color=TEXT,
                wraplength=320, justify="left", anchor="nw",
            ).grid(row=1, column=1, padx=(0, 14), pady=(8, 2), sticky="nw")

            # Driving tip row
            ctk.CTkLabel(
                info_card, text="Driving tip:",
                font=("Segoe UI", 11, "bold"), text_color=SUBTEXT, anchor="nw",
            ).grid(row=2, column=0, padx=(14, 8), pady=(4, 14), sticky="nw")
            ctk.CTkLabel(
                info_card, text=sign_data["driving_tip"],
                font=("Segoe UI", 11), text_color=TEXT,
                wraplength=320, justify="left", anchor="nw",
            ).grid(row=2, column=1, padx=(0, 14), pady=(4, 14), sticky="nw")

        # ── Disclaimer note ───────────────────────────────────────────
        ctk.CTkLabel(
            scroll,
            text="ℹ  Confidence reflects model probability, not real-world accuracy.\n"
                 "   This application is for educational purposes only.",
            font=("Segoe UI", 10), text_color=SUBTEXT, justify="left",
        ).grid(row=row_idx, column=0, padx=20, pady=(6, 20), sticky="w")

    def _make_top3_row(self, parent, row_idx, rank, label, class_id, prob, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=row_idx, column=0, sticky="ew", padx=14, pady=6)
        row.columnconfigure(2, weight=1)

        # Rank
        ctk.CTkLabel(
            row, text=f"{rank}.", width=20,
            font=("Segoe UI", 12, "bold"), text_color=color,
        ).grid(row=0, column=0, padx=(0, 8))

        # Label
        ctk.CTkLabel(
            row, text=label, anchor="w",
            font=("Segoe UI", 12), text_color=TEXT,
        ).grid(row=0, column=1, sticky="w", padx=(0, 12))

        # Bar
        bar = ctk.CTkProgressBar(
            row, fg_color=BORDER, progress_color=color, height=6, corner_radius=3
        )
        bar.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        bar.set(prob)

        # Percentage
        ctk.CTkLabel(
            row, text=f"{prob * 100:.1f}%", width=48,
            font=("Segoe UI", 11, "bold"), text_color=color, anchor="e",
        ).grid(row=0, column=3)

        ctk.CTkLabel(
            row, text=f"(ID {class_id})", width=52,
            font=("Segoe UI", 10), text_color=SUBTEXT,
        ).grid(row=0, column=4, padx=(4, 0))

    def _show_error(self, message: str) -> None:
        """Display an error state in the right panel."""
        self._clear_right()
        err_frame = ctk.CTkFrame(self._right_frame, fg_color="transparent")
        err_frame.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(err_frame, text="❌", font=("Segoe UI", 40)).pack()
        ctk.CTkLabel(
            err_frame, text="Prediction Failed",
            font=("Segoe UI", 15, "bold"), text_color=ERROR,
        ).pack(pady=(8, 4))
        ctk.CTkLabel(
            err_frame, text=message,
            font=("Segoe UI", 12), text_color=SUBTEXT,
            justify="center", wraplength=340,
        ).pack()
        ctk.CTkButton(
            err_frame, text="Try Again",
            font=("Segoe UI", 12), fg_color=ACCENT, text_color="#0B1120",
            hover_color="#0EA5E9", corner_radius=8, height=36,
            command=self._show_placeholder,
        ).pack(pady=(16, 0))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_upload(self) -> None:
        formats = " ".join(f"*{f}" for f in SUPPORTED_FORMATS)
        path = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=[
                ("Image files", formats),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _on_drop(self, event) -> None:
        """Handle drag-and-drop file event."""
        raw = event.data if hasattr(event, "data") else str(event)
        # tkinterdnd2 may wrap paths in braces on Windows
        path = raw.strip().strip("{}")
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        """Validate and load an image file; update preview."""
        # Validate
        validator = ImageValidator()
        valid, msg = validator.validate_file(path)
        if not valid:
            messagebox.showerror("Invalid Image", msg, parent=self)
            return

        try:
            pil_img = self._image_service.load_image(path)
        except ValueError as exc:
            messagebox.showerror("Image Error", str(exc), parent=self)
            return

        self._pil_image = pil_img
        self._image_path = path

        # Update file info
        info = self._image_service.get_image_info(path, pil_img)
        self._info_vals["filename"].configure(text=info["filename"])
        self._info_vals["dimensions"].configure(text=info["dimensions"])
        self._info_vals["file_size"].configure(text=f"{info['file_size_kb']} KB")
        self._info_vals["format"].configure(text=info["format"])
        self._info_frame.grid()

        # Resize for preview (max 380 × 280)
        preview = self._image_service.resize_for_display(pil_img, max_size=(380, 280))
        ctk_img = ctk.CTkImage(light_image=preview, dark_image=preview,
                               size=preview.size)
        self._preview_lbl.configure(image=ctk_img, text="")
        self._preview_lbl._image = ctk_img   # prevent GC

        # Update drop zone text
        self._drop_text_lbl.configure(text=f"✓  {Path(path).name}")
        self._drop_icon_lbl.configure(text="🖼")

        # Enable buttons
        self._predict_btn.configure(state="normal")
        self._clear_btn.configure(state="normal")

        # Show placeholder in right panel (fresh image)
        self._show_placeholder()

    def _on_clear(self) -> None:
        self._pil_image = None
        self._image_path = None
        self._info_frame.grid_remove()
        self._preview_lbl.configure(image=None, text="")
        self._drop_icon_lbl.configure(text="📁")
        self._drop_text_lbl.configure(
            text="Drop image here or click Upload\nSupported: JPG, JPEG, PNG, BMP"
        )
        self._predict_btn.configure(state="disabled")
        self._clear_btn.configure(state="disabled")
        self._show_placeholder()

    def _on_predict(self) -> None:
        if self._pil_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.", parent=self)
            return
        if self._is_predicting:
            return

        self._is_predicting = True
        self._predict_btn.configure(state="disabled", text="Predicting…")
        self._progress.grid()
        self._progress.start()

        # Snapshot values for background thread
        pil_img = self._pil_image
        filename = Path(self._image_path).name if self._image_path else "unknown"

        def _worker():
            try:
                result = self._prediction_service.predict(pil_img)
                result["timestamp"] = datetime.now().isoformat()
                result["filename"] = filename
                self.after(0, lambda: self._on_predict_done(result, filename))
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda: self._on_predict_error(err_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_predict_done(self, result: dict, filename: str) -> None:
        """Called on main thread after prediction succeeds."""
        self._is_predicting = False
        self._progress.stop()
        self._progress.grid_remove()
        self._predict_btn.configure(state="normal", text="🔍  Predict Sign")

        # Save to history
        try:
            self._history_service.add_record(result)
        except Exception:
            pass

        self._show_result(result, filename)

    def _on_predict_error(self, message: str) -> None:
        """Called on main thread when prediction fails."""
        self._is_predicting = False
        self._progress.stop()
        self._progress.grid_remove()
        self._predict_btn.configure(state="normal", text="🔍  Predict Sign")
        self._show_error(message)

    # ------------------------------------------------------------------
    # Refresh (called by App.show_page)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """No-op — state preserved between visits."""
        pass
