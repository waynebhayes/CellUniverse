from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import tifffile as tiff
import imageio.v2 as imageio
import numpy as np
import argparse
import re

# ---------- command line arguments ----------
parser = argparse.ArgumentParser(
    description="Merge PNG stacks into TIFF"
)

parser.add_argument("--i", required=True,
                    help="Input root folder")

parser.add_argument("--o", required=True,
                    help="Output folder")

parser.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers (0 = os.cpu_count())")

args = parser.parse_args()

input_root = Path(args.i)
output_root = Path(args.o)
output_root.mkdir(parents=True, exist_ok=True)

def numeric_filename_key(path: Path):
    # Natural sort by numeric chunks in filename (e.g., img2 before img10).
    return [int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", path.stem)]


def convert_folder(folder: Path) -> str:
    pngs = sorted(folder.glob("*.png"), key=numeric_filename_key)
    if not pngs:
        return ""
    # Read PNGs in parallel within the folder too — each frame is ~225 PNGs
    # and imageio.imread releases the GIL during decode.
    with ThreadPoolExecutor(max_workers=min(8, len(pngs))) as inner:
        imgs = list(inner.map(imageio.imread, pngs))
    stack = np.stack(imgs)
    out_file = output_root / f"{folder.name}.tiff"
    tiff.imwrite(out_file, stack)
    return str(out_file)


# ---------- processing ----------
# Case A: PNGs directly under root (no subfolders)
root_pngs = sorted(input_root.glob("*.png"), key=numeric_filename_key)
if root_pngs:
    with ThreadPoolExecutor(max_workers=min(8, len(root_pngs))) as inner:
        imgs = list(inner.map(imageio.imread, root_pngs))
    stack = np.stack(imgs)
    out_file = output_root / f"{input_root.name}.tiff"
    tiff.imwrite(out_file, stack)
    print("Saved:", out_file)

# Case B: PNGs inside subfolders — convert folders in parallel.
folders = [f for f in input_root.iterdir() if f.is_dir()]
if folders:
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=min(n_workers, len(folders))) as outer:
        for out_path in outer.map(convert_folder, folders):
            if out_path:
                print("Saved:", out_path)
