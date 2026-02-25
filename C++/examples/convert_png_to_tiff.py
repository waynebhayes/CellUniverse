from pathlib import Path
import tifffile as tiff
import imageio.v2 as imageio
import numpy as np
import argparse

# ---------- command line arguments ----------
parser = argparse.ArgumentParser(
    description="Merge PNG stacks into TIFF"
)

parser.add_argument("--i", required=True,
                    help="Input root folder")

parser.add_argument("--o", required=True,
                    help="Output folder")

args = parser.parse_args()

input_root = Path(args.i)
output_root = Path(args.o)
output_root.mkdir(parents=True, exist_ok=True)

# ---------- processing ----------
# Case A: PNGs directly under root (no subfolders)
root_pngs = sorted(input_root.glob("*.png"))
if root_pngs:
    imgs = [imageio.imread(f) for f in root_pngs]
    stack = np.stack(imgs)

    out_file = output_root / f"{input_root.name}.tiff"
    tiff.imwrite(out_file, stack)
    print("Saved:", out_file)

# Case B: PNGs inside subfolders
for folder in input_root.iterdir():
    if not folder.is_dir():
        continue

    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        continue

    imgs = [imageio.imread(f) for f in pngs]
    stack = np.stack(imgs)

    out_file = output_root / f"{folder.name}.tiff"
    tiff.imwrite(out_file, stack)

    print("Saved:", out_file)