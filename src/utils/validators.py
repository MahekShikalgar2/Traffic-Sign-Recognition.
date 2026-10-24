"""
Validation utilities for image files and PIL Image objects.
"""

import os
from pathlib import Path
from typing import Tuple

from src.config import SUPPORTED_FORMATS, MAX_FILE_SIZE_MB


class ImageValidator:
    """Static validation helpers for image files and PIL Image objects."""

    @staticmethod
    def validate_file(path: str) -> Tuple[bool, str]:
        """
        Validate that *path* is a readable image file.

        Returns
        -------
        (True, '')            on success
        (False, error_msg)    on failure
        """
        if not path:
            return False, "No file path provided."

        p = Path(path)

        if not p.exists():
            return False, f"File not found: {p.name}"

        if not p.is_file():
            return False, f"Path is not a file: {p.name}"

        ext = p.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return False, (
                f"Unsupported format '{ext}'. "
                f"Accepted formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        size_bytes = p.stat().st_size
        if size_bytes == 0:
            return False, "The file is empty (0 bytes)."

        size_mb = size_bytes / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, (
                f"File is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            )

        return True, ""

    @staticmethod
    def validate_image(pil_image) -> Tuple[bool, str]:
        """
        Validate a PIL Image object.

        Returns
        -------
        (True, '')            on success
        (False, error_msg)    on failure
        """
        if pil_image is None:
            return False, "Image object is None."

        try:
            width, height = pil_image.size
        except Exception as exc:
            return False, f"Cannot read image dimensions: {exc}"

        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions: {width}×{height}."

        valid_modes = {'L', 'RGB', 'RGBA', 'P', 'I', 'F', 'LA', 'PA',
                       'CMYK', 'YCbCr', 'LAB', 'HSV', 'RGBa', 'La',
                       '1', 'BGR;15', 'BGR;16', 'BGR;24'}
        if pil_image.mode not in valid_modes:
            # Don't block — just warn in calling code; model converts to 'L'
            pass

        return True, ""
