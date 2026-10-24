"""
AboutPage — project info, tech stack, dataset citation, and disclaimer.
"""

import customtkinter as ctk

from src.config import (
    CARD, ACCENT, ACCENT2, SUCCESS, WARNING, ERROR, TEXT, SUBTEXT, BORDER, CONTENT,
)


class AboutPage(ctk.CTkFrame):
    """About page: app info, stack, dataset, disclaimer."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, fg_color=CONTENT, **kwargs)
        self._app = app
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent", scrollbar_button_color=BORDER,
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=30, pady=20)
        scroll.columnconfigure(0, weight=1)

        row_i = 0

        # ── App hero ─────────────────────────────────────────────────────
        hero = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=14)
        hero.grid(row=row_i, column=0, sticky="ew", pady=(0, 16))
        hero.columnconfigure(0, weight=1)
        row_i += 1

        ctk.CTkLabel(hero, text="🚦", font=("Segoe UI", 52)).grid(
            row=0, column=0, pady=(28, 6)
        )
        ctk.CTkLabel(
            hero, text="Traffic Sign Recognition",
            font=("Segoe UI", 26, "bold"), text_color=TEXT,
        ).grid(row=1, column=0, pady=(0, 4))
        ctk.CTkLabel(
            hero, text="AI-Powered Computer Vision Dashboard",
            font=("Segoe UI", 14), text_color=SUBTEXT,
        ).grid(row=2, column=0)

        badges = ctk.CTkFrame(hero, fg_color="transparent")
        badges.grid(row=3, column=0, pady=(10, 22))

        for text, color in [
            ("v1.0.0", ACCENT),
            ("Python 3.11", ACCENT2),
            ("TensorFlow 2.15", SUCCESS),
            ("43 Classes", WARNING),
        ]:
            badge = ctk.CTkFrame(badges, fg_color=CARD, corner_radius=20, border_width=1, border_color=color)
            badge.pack(side="left", padx=5)
            ctk.CTkLabel(badge, text=text, font=("Segoe UI", 11), text_color=color).pack(padx=12, pady=4)

        # ── Description ───────────────────────────────────────────────────
        row_i = self._section(scroll, row_i, "About This Project")
        desc_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        desc_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 14))
        row_i += 1
        ctk.CTkLabel(
            desc_card,
            text=(
                "This application classifies German traffic signs using a Convolutional Neural Network (CNN)\n"
                "trained on the GTSRB dataset. It provides an interactive desktop interface for uploading\n"
                "images, running inference, and reviewing prediction history.\n\n"
                "The model processes 32×32 grayscale images and outputs probabilities across 43 sign classes."
            ),
            font=("Segoe UI", 12), text_color=TEXT,
            justify="left", wraplength=800,
        ).grid(padx=20, pady=16, sticky="w")

        # ── Technology stack ──────────────────────────────────────────────
        row_i = self._section(scroll, row_i, "Technology Stack")

        stack_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        stack_frame.grid(row=row_i, column=0, sticky="ew", pady=(0, 14))
        for i in range(3):
            stack_frame.columnconfigure(i, weight=1)
        row_i += 1

        stack_items = [
            ("🐍", "Python", "3.11.9", "Core language", ACCENT),
            ("🤖", "TensorFlow", "2.15.1", "Deep learning framework", ACCENT2),
            ("⚙️", "Keras", "2.15.0", "Model API", SUCCESS),
            ("🖥️", "CustomTkinter", "5.2.2", "Modern UI framework", WARNING),
            ("🔢", "NumPy", "1.26.4", "Numerical computing", ACCENT),
            ("🖼️", "Pillow", "10.4.0", "Image processing", ACCENT2),
        ]

        for s_i, (icon, name, version, desc, color) in enumerate(stack_items):
            c_row, c_col = divmod(s_i, 3)
            card = ctk.CTkFrame(stack_frame, fg_color=CARD, corner_radius=10)
            card.grid(
                row=c_row, column=c_col, sticky="ew",
                padx=(0, 10) if c_col < 2 else 0, pady=(0, 8),
            )
            ctk.CTkLabel(card, text=icon, font=("Segoe UI", 24)).grid(
                row=0, column=0, padx=16, pady=(14, 4), sticky="w"
            )
            ctk.CTkLabel(
                card, text=f"{name}  {version}",
                font=("Segoe UI", 13, "bold"), text_color=color,
            ).grid(row=1, column=0, padx=16, sticky="w")
            ctk.CTkLabel(
                card, text=desc, font=("Segoe UI", 10), text_color=SUBTEXT,
            ).grid(row=2, column=0, padx=16, pady=(2, 14), sticky="w")

        # ── Dataset info ──────────────────────────────────────────────────
        row_i = self._section(scroll, row_i, "Dataset")

        ds_card = ctk.CTkFrame(scroll, fg_color=CARD, corner_radius=10)
        ds_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 14))
        ds_card.columnconfigure(0, weight=1)
        row_i += 1

        ds_fields = [
            ("Name", "German Traffic Sign Recognition Benchmark (GTSRB)"),
            ("Total Images", "34,776"),
            ("Classes", "43 traffic sign categories (class IDs 1–43)"),
            ("Source", "Institut für Neuroinformatik, Ruhr-Universität Bochum"),
            ("License", "Freely available for research and educational use"),
        ]
        for d_i, (k, v) in enumerate(ds_fields):
            bg = CARD if d_i % 2 == 0 else "#1A2744"
            d_row = ctk.CTkFrame(ds_card, fg_color=bg, corner_radius=0)
            d_row.grid(row=d_i, column=0, sticky="ew")
            d_row.columnconfigure(1, weight=1)
            ctk.CTkLabel(
                d_row, text=k, width=150,
                font=("Segoe UI", 11), text_color=SUBTEXT, anchor="w",
            ).grid(row=0, column=0, padx=(14, 6), pady=8, sticky="w")
            ctk.CTkLabel(
                d_row, text=v, anchor="w",
                font=("Segoe UI", 11), text_color=TEXT,
            ).grid(row=0, column=1, padx=(0, 14), pady=8, sticky="w")

        # ── Disclaimer ────────────────────────────────────────────────────
        row_i = self._section(scroll, row_i, "Disclaimer")

        disc_card = ctk.CTkFrame(scroll, fg_color="#172033", corner_radius=10,
                                  border_width=1, border_color=WARNING)
        disc_card.grid(row=row_i, column=0, sticky="ew", pady=(0, 20))
        disc_card.columnconfigure(0, weight=1)
        row_i += 1

        ctk.CTkLabel(
            disc_card,
            text=(
                "⚠  This application is for educational and research purposes only.\n\n"
                "Do not use this model for real-world traffic safety decisions without proper validation.\n"
                "Model confidence scores reflect learned training patterns and are not a guarantee of\n"
                "real-world accuracy. Always consult certified traffic safety systems for critical use."
            ),
            font=("Segoe UI", 11), text_color=TEXT,
            justify="left", wraplength=800,
        ).grid(padx=20, pady=16, sticky="w")

    def _section(self, parent, row_i, title) -> int:
        ctk.CTkLabel(
            parent, text=title,
            font=("Segoe UI", 15, "bold"), text_color=TEXT,
        ).grid(row=row_i, column=0, sticky="w", pady=(4, 8))
        return row_i + 1

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        pass
