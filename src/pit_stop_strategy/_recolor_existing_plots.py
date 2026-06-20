"""One-off utility: convert previously-generated dark-themed pit-stop PNGs
to a clean white-themed equivalent without re-running the full pipeline
(the raw FastF1 telemetry CSV is not shipped with the repo).

Strategy: target the exact known dark palette used by visualize.py and
remap only those pixels.

  BG     #0F0F1A  page background       -> #FFFFFF white
  PANEL  #1A1A2E  axes background       -> #FFFFFF white
  GRID   #252540  grid lines            -> #D7D7DC light gray
  BORDER #333355  axis spines / boxes   -> #888888 medium gray
  TEXT   #E8E8F0  near-white labels     -> #222222 dark text
  MUTED  #8888AA  tick labels           -> #555555 dark gray text
  F1_WHITE #F5F5F5 line/marker over BG  -> #222222 dark gray
  legend #22224A  legend background     -> #FFFFFF white

Mapping uses RGB nearest-neighbour with a tolerance, then snaps each
matched pixel to the new palette colour while preserving alpha.  Saturated
colours (brand reds, blues, teals, oranges) sit far outside this set and
are left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "outputs" / "pit_stop_strategy" / "plots"

# (source_rgb, target_rgb, tol_in_rgb_distance)
COLOR_MAP = [
    ((0x0F, 0x0F, 0x1A), (255, 255, 255), 18),  # BG -> white
    ((0x1A, 0x1A, 0x2E), (255, 255, 255), 18),  # PANEL -> white
    ((0x22, 0x22, 0x4A), (255, 255, 255), 22),  # legend bg -> white
    ((0x25, 0x25, 0x40), (0xD7, 0xD7, 0xDC), 18),  # GRID -> light gray
    ((0x33, 0x33, 0x55), (0x88, 0x88, 0x88), 22),  # BORDER -> medium gray
    ((0xE8, 0xE8, 0xF0), (0x22, 0x22, 0x22), 14),  # TEXT -> dark
    ((0xF5, 0xF5, 0xF5), (0x22, 0x22, 0x22), 14),  # F1_WHITE -> dark
    ((0x88, 0x88, 0xAA), (0x55, 0x55, 0x55), 16),  # MUTED -> dark gray
]


def _remap(rgb: np.ndarray) -> np.ndarray:
    """Apply the COLOR_MAP in order; first hit wins."""
    out = rgb.copy()
    matched = np.zeros(rgb.shape[:2], dtype=bool)

    for src, tgt, tol in COLOR_MAP:
        src_arr = np.array(src, dtype=np.int16)
        diff = rgb.astype(np.int16) - src_arr
        dist = np.sqrt((diff * diff).sum(axis=-1))
        m = (dist <= tol) & (~matched)
        if not m.any():
            continue
        # Linear blend: the further from src, the less of tgt to apply
        # (so antialiased pixels around text/lines transition smoothly).
        weight = np.clip(1.0 - dist / max(tol, 1), 0.0, 1.0)[..., None]
        out_f = (np.array(tgt, dtype=np.float32)[None, None, :] * weight
                 + out.astype(np.float32) * (1.0 - weight))
        out[m] = np.clip(out_f[m], 0, 255).astype(np.uint8)
        matched |= m

    return out


def recolor_image(in_path: Path, out_path: Path) -> None:
    im = Image.open(in_path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]

    new_rgb = _remap(rgb)

    # Crisp white sweep: any pixel that ended up within 5 of pure white
    # snaps to pure white.  Prevents the faint navy tint left behind by
    # antialias halos.
    is_near_white = ((255 - new_rgb).max(axis=-1) <= 8)
    new_rgb[is_near_white] = 255

    out_arr = np.concatenate([new_rgb, alpha], axis=-1)
    Image.fromarray(out_arr).save(out_path, optimize=True)
    print(f"  [recolored] {in_path.name}")


def main() -> None:
    if not PLOTS_DIR.exists():
        print(f"No plots directory: {PLOTS_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(PLOTS_DIR.glob("*.png"))
    if not files:
        print("No PNGs found.", file=sys.stderr)
        sys.exit(1)

    for p in files:
        recolor_image(p, p)

    print(f"\nDone. {len(files)} file(s) recolored under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
