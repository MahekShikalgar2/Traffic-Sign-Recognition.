"""
ImageService — responsible for loading and inspecting PIL Images.
"""

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageService:
    """Helper service for image loading and metadata extraction."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: str) -> Image.Image:
        """
        Open and return a PIL Image from *path*.

        Raises
        ------
        ValueError
            With a user-friendly message if the file cannot be opened
            (corrupt file, unsupported format, I/O error, etc.).
        """
        try:
            img = Image.open(path)
            # Force-load pixel data now to catch truncated / corrupt files
            img.load()
            return img
        except FileNotFoundError:
            raise ValueError(f"File not found: {Path(path).name}")
        except UnidentifiedImageError:
            raise ValueError(
                f"Cannot identify image file '{Path(path).name}'. "
                "The file may be corrupt or in an unsupported format."
            )
        except OSError as exc:
            raise ValueError(
                f"Could not open image '{Path(path).name}': {exc}"
            )
        except Exception as exc:
            raise ValueError(
                f"Unexpected error loading '{Path(path).name}': {exc}"
            )

    @staticmethod
    def get_image_info(path: str, pil_image: Image.Image) -> dict:
        """
        Return metadata about an image.

        Returns
        -------
        dict with keys:
            filename      str   e.g. "stop_sign.jpg"
            filepath      str   absolute path
            dimensions    str   e.g. "640×480"
            file_size_kb  float
            mode          str   e.g. "RGB"
            format        str   e.g. "JPEG"  (may be None for in-memory images)
        """
        p = Path(path)
        width, height = pil_image.size
        size_kb = round(p.stat().st_size / 1024, 1) if p.exists() else 0.0

        return {
            "filename":     p.name,
            "filepath":     str(p.resolve()),
            "dimensions":   f"{width}×{height}",
            "file_size_kb": size_kb,
            "mode":         pil_image.mode,
            "format":       pil_image.format or "Unknown",
        }

    @staticmethod
    def resize_for_display(pil_image: Image.Image,
                           max_size: tuple = (400, 400)) -> Image.Image:
        """
        Return a copy of *pil_image* scaled to fit within *max_size*,
        preserving aspect ratio.
        """
        img = pil_image.copy()
        img.thumbnail(max_size, Image.LANCZOS)
        return img
