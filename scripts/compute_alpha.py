#!/usr/bin/env python3
"""process_alpha_matte.py – Safe wrapper for Closed-Form Matting
----------------------------------------------------------------
• Accepts image + scribbles, handles size and memory safely.
• Automatically rescales if the image exceeds a pixel threshold or max dimension.
• Asserts that scribbles and image match dimensions (with optional 90° auto-rotation).
• Uses logging instead of prints and argparse for CLI arguments.
• Always saves the alpha matte; no display windows are used.

Quick usage:
~~~~~~~~~~~~
$ python process_alpha_matte.py image.png scribbles.png -m 500 -o alpha.png

Dependencies: numpy, opencv-python, closed_form_matting

Authors: Davide Franchi, Biaggio Cancelliere, Carlos Ruiz.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import closed_form_matting

# Global safety parameter
MAX_SAFE_PIXELS = 4_000_000  # 4 MP = 3000×1333, avoids running out of RAM

def _read_image(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk or raise an error."""
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def _fix_orientation(ref: np.ndarray, scrib: np.ndarray) -> np.ndarray:
    """
    Ensure scribbles are 3-channel BGR and oriented to match ref.
    If scrib is single-channel, convert to BGR.
    If scrib appears rotated 90°, rotate it back.
    Finally, assert that shapes match.
    """
    # Convert grayscale scribbles to BGR
    if scrib.ndim == 2:
        logging.warning("Scribbles are single-channel; converting to BGR.")
        scrib = cv2.cvtColor(scrib, cv2.COLOR_GRAY2BGR)

    # Detect if height and width are swapped (rotated 90°)
    if scrib.shape[0] == ref.shape[1] and scrib.shape[1] == ref.shape[0]:
        logging.warning("Scribbles appear rotated 90°; rotating to match image orientation.")
        scrib = cv2.rotate(scrib, cv2.ROTATE_90_CLOCKWISE)

    # Assert final shape matches
    if scrib.shape != ref.shape:
        raise ValueError(f"Scribbles shape {scrib.shape} does not match image {ref.shape} after auto-fix.")
    return scrib

def _resize_pair(img: np.ndarray, scrib: np.ndarray, max_dim: int | None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    If max_dim is not None and either dimension exceeds max_dim,
    compute a scale factor and resize both img and scrib to that scale,
    preserving aspect ratio. Returns (img_resized, scrib_resized, scale_factor).
    """
    h, w = img.shape[:2]
    if max_dim is None or max(h, w) <= max_dim:
        return img, scrib, 1.0

    scale = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    logging.info("Resizing from %dx%d to %dx%d to reduce memory use.", w, h, *new_size)
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    scrib_resized = cv2.resize(scrib, new_size, interpolation=cv2.INTER_AREA)
    return img_resized, scrib_resized, scale

def _compute_alpha(img_bgr: np.ndarray, scrib_bgr: np.ndarray) -> np.ndarray:
    """
    Call closed_form_matting on normalized floats in [0, 1].
    Returns a single-channel float32 alpha matte in [0, 1].
    """
    img_f32 = img_bgr.astype(np.float32) / 255.0
    scrib_f32 = scrib_bgr.astype(np.float32) / 255.0
    return closed_form_matting.closed_form_matting_with_scribbles(img_f32, scrib_f32)

def process_alpha_matte(image_path: str,
                        scribbles_path: str,
                        max_dimension: int | None = 500,
                        output: str | os.PathLike = "alpha.png") -> np.ndarray:
    """
    Compute and save the alpha matte using closed-form matting.

    Args:
        image_path (str): Path to the input image.
        scribbles_path (str): Path to the scribbles image.
        max_dimension (int | None): Max dimension to rescale if necessary (0 or None to disable).
        output (str | Path): Where to save the output alpha matte.

    Returns:
        np.ndarray: Alpha matte in [0, 1] float32 range, resized to original size.
    """
    # Read both images
    img = _read_image(image_path)
    scrib = _read_image(scribbles_path)

    # If total pixels exceed threshold and no max_dimension was set, force a downscale
    pixels = img.shape[0] * img.shape[1]
    if pixels > MAX_SAFE_PIXELS and max_dimension is None:
        logging.warning("Image has %d pixels (> %d). Forcing downscale to max_dim=500.", pixels, MAX_SAFE_PIXELS)
        max_dimension = 500

    # Ensure scribbles match orientation and channels
    scrib = _fix_orientation(img, scrib)

    # Resize both if needed
    img_s, scrib_s, scale = _resize_pair(img, scrib, max_dimension)

    logging.info("Computing alpha matte...")
    alpha = _compute_alpha(img_s, scrib_s)

    # If we resized for processing, bring alpha back to original size
    if scale != 1.0:
        alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Prepare output path
    out_path = Path(output)
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save alpha as 8-bit PNG
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), alpha_uint8)
    logging.info("Alpha matte saved to %s", out_path)

    return alpha

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-Form Matting (safe wrapper, no display windows)"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("scribbles", help="Path to the scribbles image (same shape as input)")
    parser.add_argument(
        "-m", "--max-dim", type=int, default=500,
        help="Maximum dimension allowed before resizing (0 to disable)"
    )
    parser.add_argument(
        "-o", "--output", default="alpha.png",
        help="Path to output PNG alpha matte"
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    max_dim = None if args.max_dim == 0 else args.max_dim
    process_alpha_matte(args.image, args.scribbles, max_dimension=max_dim, output=args.output)

if __name__ == "__main__":
    _cli()
