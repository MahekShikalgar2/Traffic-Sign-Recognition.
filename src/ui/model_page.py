"""
ModelPage — displays CNN model architecture info from ModelService.
"""

import customtkinter as ctk

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, TEXT, SUBTEXT, BORDER, CONTENT,
)


class ModelPage(ctk.CTkFrame):
    """Model information page showing architecture, params, and training notes."""

    def __init__(self, master, app, model_service, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._model_service = model_service
        self._built = False
        self._build_static_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_static_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Header ──────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(28, 0))
        ctk.CTkLabel(
            header, text="Model Information",
            font=("Segoe UI", 24, "bold"), text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="CNN architecture trained on GTSRB dataset",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Scrollable content ───────────────────────────────────────────
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=30, pady=14)
        self._scroll.columnconfigure(0, weight=1)
        self._scroll.columnconfigure(1, weight=1)

        # Placeholder until model info is loaded
        self._placeholder = ctk.CTkLabel(
            self._scroll, text="Loading model information…",
            font=("Segoe UI", 13), text_color=SUBTEXT,
        )
        self._placeholder.grid(row=0, column=0, columnspan=2, pady=40)

    def _build_model_content(self, info: dict) -> None:
        """Populate scroll area with actual model data (called once)."""
        if self._built:
            return
        self._built = True
        self._placeholder.destroy()

        row_i = 0
        scroll = self._scroll

        # ── Info cards ────────────────────────────────────────────────
        cards_data = [
            ("📥", "Input Shape", str(info.get("input_shape", "—")), ACCENT),
            ("📤", "Output Shape", str(info.get("output_shape", "—")), ACCENT2),
            ("🔢", "Total Params", f"{info.get('total_params', 0):,}", SUCCESS),
            ("📁", "File Size", f"{info.get('file_size_mb', 0)} MB", WARNING),
            ("🧱", "Layers", str(info.get("num_layers", "—")), ACCENT),
            ("🤖", "Framework", "TensorFlow / Keras", ACCENT2),
        ]

        cards_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        cards_frame.grid(row=row_i, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        for i in range(3):
            cards_frame.columnconfigure(i, weight=1)
        row_i += 1

        for card_i, (icon, title, val, color) in enumerate(cards_data):
            c_row, c_col = divmod(card_i, 3)
            card = ctk.CTkFrame(cards_frame, fg_color=CARD, corner_radius=10)
            card.grid(
                row=c_row, column=c_col, sticky="ew",
                padx=(0, 10) if c_col < 2 else 0, pady=(0, 8),
            )
            card.columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=icon, font=("Segoe UI", 20)).grid(
                row=0, column=0, padx=16, pady=(14, 4), sticky="w"
            )
            ctk.CTkLabel(
                card, text=val, font=("Segoe UI", 15, "bold"), text_color=color
            ).grid(row=1, column=0, padx=16, sticky="w")
            ctk.CTkLabel(
                card, text=title, font=("Segoe UI", 10), text_color=SUBTEXT
            ).grid(row=2, column=0, padx=16, pady=(2, 14), sticky="w")

        # ── Layer table ───────────────────────────────────────────────
        ctk.CTkLabel(
            scroll, text="Layer Architecture",
            font=("Segoe UI", 14, "bold"), text_color=TEXT,
        ).grid(row=row_i, column=0, columnspan=2, sticky="w", pady=(4, 8))
        row_i += 1

        table_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        table_card.grid(row=row_i, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        table_card.columnconfigure(1, weight=1)
        row_i += 1

        # Table header
        hdr = ctk.CTkFrame(table_card, fg_color="#243047", corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=4, sticky="ew")
        for col_i, (h_text, w) in enumerate([("Layer", 160), ("Type", 0), ("Output Shape", 180), ("Params", 90)]):
            kw = {"width": w} if w else {}
            ctk.CTkLabel(
                hdr, text=h_text, font=("Segoe UI", 11, "bold"), text_color=SUBTEXT,
                anchor="w", **kw,
            ).grid(row=0, column=col_i, padx=(14 if col_i == 0 else 6, 6), pady=8, sticky="w")
            if not w:
                hdr.columnconfigure(col_i, weight=1)

        # Layer rows from model
        try:
            from keras.models import load_model as _km
            model = self._model_service._model
            if model is None:
                raise RuntimeError("Model not loaded")
            for r_i, layer in enumerate(model.layers):
                bg = CARD if r_i % 2 == 0 else "#1A2744"
                l_row = ctk.CTkFrame(table_card, fg_color=bg, corner_radius=0)
                l_row.grid(row=r_i + 1, column=0, columnspan=4, sticky="ew")
                l_row.columnconfigure(1, weight=1)

                try:
                    out_shape = str(layer.output_shape)
                except Exception:
                    out_shape = "—"
                try:
                    n_params = f"{layer.count_params():,}"
                except Exception:
                    n_params = "—"

                layer_cells = [
                    (layer.name, 160, ACCENT, "w"),
                    (layer.__class__.__name__, 0, TEXT, "w"),
                    (out_shape, 180, SUBTEXT, "w"),
                    (n_params, 90, TEXT, "e"),
                ]
                for c_i, (text, width, color, anchor) in enumerate(layer_cells):
                    kw = {"width": width} if width else {}
                    ctk.CTkLabel(
                        l_row, text=text, font=("Segoe UI", 10),
                        text_color=color, anchor=anchor, **kw,
                    ).grid(
                        row=0, column=c_i,
                        padx=(14 if c_i == 0 else 6, 6), pady=6, sticky="w",
                    )
                    if not width:
                        l_row.columnconfigure(c_i, weight=1)
        except Exception as exc:
            ctk.CTkLabel(
                table_card, text=f"Could not enumerate layers: {exc}",
                font=("Segoe UI", 11), text_color=SUBTEXT,
            ).grid(row=1, column=0, columnspan=4, padx=16, pady=12, sticky="w")

        # ── Training info ──────────────────────────────────────────────
        ctk.CTkLabel(
            scroll, text="Training Details",
            font=("Segoe UI", 14, "bold"), text_color=TEXT,
        ).grid(row=row_i, column=0, columnspan=2, sticky="w", pady=(4, 8))
        row_i += 1

        training_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        training_card.grid(row=row_i, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        training_card.columnconfigure(1, weight=1)
        row_i += 1

        training_fields = [
            ("Dataset", "GTSRB (German Traffic Sign Recognition Benchmark)"),
            ("Total Images", "34,776"),
            ("Classes", "43 traffic sign categories"),
            ("Image Input", "32 × 32 grayscale (1 channel)"),
            ("Preprocessing", "Convert to grayscale → resize → normalize [0, 1]"),
            ("Optimizer", "Adam"),
            ("Loss Function", "Categorical Cross-Entropy"),
            ("Training Accuracy", "~96% (training split — not real-world accuracy)"),
        ]
        for t_i, (key, value) in enumerate(training_fields):
            bg = CARD if t_i % 2 == 0 else "#1A2744"
            t_row = ctk.CTkFrame(training_card, fg_color=bg, corner_radius=0)
            t_row.grid(row=t_i, column=0, columnspan=2, sticky="ew")
            t_row.columnconfigure(1, weight=1)
            ctk.CTkLabel(
                t_row, text=key, width=160,
                font=("Segoe UI", 11), text_color=SUBTEXT, anchor="w",
            ).grid(row=0, column=0, padx=(14, 6), pady=8, sticky="w")
            ctk.CTkLabel(
                t_row, text=value, anchor="w",
                font=("Segoe UI", 11), text_color=TEXT,
            ).grid(row=0, column=1, padx=(0, 14), pady=8, sticky="w")

        # ── Performance disclaimer ─────────────────────────────────────
        warn_frame = ctk.CTkFrame(scroll, fg_color="#451A03", corner_radius=8)
        warn_frame.grid(row=row_i, column=0, columnspan=2, sticky="ew", pady=(4, 20))
        warn_frame.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            warn_frame,
            text="⚠  Performance Note",
            font=("Segoe UI", 12, "bold"), text_color=WARNING,
        ).grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")
        ctk.CTkLabel(
            warn_frame,
            text="Accuracy figures shown are from model training and validation.\n"
                 "Real-world performance depends on image quality, lighting, angle, and other factors.\n"
                 "Do not rely on this model for real traffic safety decisions.",
            font=("Segoe UI", 11), text_color=TEXT, justify="left", wraplength=700,
        ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if not self._built and self._model_service.is_loaded:
            try:
                info = self._model_service.get_model_info()
                self._build_model_content(info)
            except Exception as exc:
                self._placeholder.configure(text=f"Error loading model info: {exc}")
