"""
Generate a self-contained HTML grid of first/last frames for all eval-episode videos.

Usage (offline / CLI):
    python src/make_eval_grid.py <run_dir>

The generated file (eval_grid.html) is placed alongside final_eval.json in run_dir.
Each row corresponds to one eval episode; the two columns show the first and last
frame of that episode's video.

Requires: imageio[pyav]  (listed in requirements.txt)
"""

from __future__ import annotations

import base64
import re
import sys
import warnings
from pathlib import Path

import imageio.v3 as iio

VIDEOS_DIR = "videos"
OUTPUT_FILENAME = "eval_grid.html"
VIDEO_PREFIX = "final_eval"


def _read_frames(video_path: Path) -> tuple[bytes, bytes]:
    """Return (first_png, last_png) bytes for *video_path*."""
    frames = iio.imread(str(video_path), plugin="pyav")  # (T, H, W, 3) uint8
    first_png: bytes = iio.imwrite("<bytes>", frames[0], extension=".png")
    last_png: bytes = iio.imwrite("<bytes>", frames[-1], extension=".png")
    return first_png, last_png


def _b64_img_tag(png_bytes: bytes, alt: str = "") -> str:
    b64 = base64.b64encode(png_bytes).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def _episode_number(path: Path) -> int:
    m = re.search(r"-episode-(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def make_eval_grid(run_dir: Path) -> Path | None:
    """
    Build eval_grid.html in *run_dir* from the videos in run_dir/videos/.

    Returns the output path on success, or None if no matching videos were found.
    Errors per video are reported as warnings so a failed frame never crashes the
    enclosing training run.
    """
    run_dir = Path(run_dir)
    video_dir = run_dir / VIDEOS_DIR

    if not video_dir.is_dir():
        warnings.warn(
            f"make_eval_grid: videos directory not found: {video_dir}"
        )
        return None

    videos = sorted(
        video_dir.glob(f"{VIDEO_PREFIX}-episode-*.mp4"),
        key=_episode_number,
    )
    if not videos:
        warnings.warn(f"make_eval_grid: no matching videos in {video_dir}")
        return None

    rows_html: list[str] = []
    for vp in videos:
        ep = _episode_number(vp)
        try:
            first_png, last_png = _read_frames(vp)
            first_cell = (
                f"<td>{_b64_img_tag(first_png, alt=f'ep {ep} first')}</td>"
            )
            last_cell = (
                f"<td>{_b64_img_tag(last_png, alt=f'ep {ep} last')}</td>"
            )
        except Exception as exc:
            warnings.warn(f"make_eval_grid: {vp.name} failed: {exc}")
            err = "<td><span style='color:#f66'>(error)</span></td>"
            first_cell = last_cell = err
        rows_html.append(f"<tr><td>{ep}</td>{first_cell}{last_cell}</tr>")

    rows_joined = "\n        ".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Grid — {run_dir.name}</title>
<style>
  body  {{ background: #1a1a1a; color: #ddd; font-family: monospace; padding: 1em; }}
  h1   {{ font-size: 1em; color: #aaa; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #444; padding: 4px 8px; text-align: center;
            vertical-align: middle; }}
  th   {{ background: #2a2a2a; color: #bbb; font-weight: normal; }}
  td:first-child {{ color: #888; min-width: 2.5em; }}
  img  {{ display: block; width: 200px; height: auto; image-rendering: pixelated; }}
</style>
</head>
<body>
<h1>{run_dir.name}</h1>
<table>
  <thead>
    <tr><th>ep</th><th>first frame</th><th>last frame</th></tr>
  </thead>
  <tbody>
        {rows_joined}
  </tbody>
</table>
</body>
</html>
"""

    out_path = run_dir / OUTPUT_FILENAME
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"Usage: python {Path(__file__).name} <run_dir>", file=sys.stderr
        )
        sys.exit(1)

    result = make_eval_grid(Path(sys.argv[1]))
    if result is None:
        sys.exit(1)
    print(f"Wrote {result}")
