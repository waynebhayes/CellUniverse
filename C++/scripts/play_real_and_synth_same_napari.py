from pathlib import Path
import re
import numpy as np
import imageio.v3 as iio
import tifffile as tiff
import napari
from qtpy.QtCore import QTimer

# =========================================================
# 路径
# =========================================================
BASE_DIR = Path("/Users/wangyiding/CellUniverse/C++/output/✅40Frames_Embryo_20260414_022211")
REAL_DIR = BASE_DIR / "real"
SYNTH_DIR = BASE_DIR / "synth"

# =========================================================
# 工具函数
# =========================================================
def natural_key(s):
    s = str(s)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]

def collect_frame_dirs(folder: Path):
    frame_dirs = [p for p in folder.iterdir() if p.is_dir()]
    frame_dirs = sorted(frame_dirs, key=lambda p: natural_key(p.name))
    return frame_dirs

def collect_recursive_tifs(folder: Path):
    files = sorted(
        list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff")),
        key=lambda p: natural_key(p.name)
    )
    return files

def load_synth_from_png_folders(folder: Path):
    frame_dirs = collect_frame_dirs(folder)
    if not frame_dirs:
        raise RuntimeError(f"[ERROR] No frame folders found under synth dir: {folder}")

    print("[INFO] synth frame folders found:")
    for p in frame_dirs[:10]:
        print("   ", p.name)
    if len(frame_dirs) > 10:
        print(f"   ... total = {len(frame_dirs)}")

    volumes = []
    expected_shape = None

    for frame_dir in frame_dirs:
        pngs = sorted(list(frame_dir.glob("*.png")), key=lambda p: natural_key(p.name))
        if not pngs:
            print(f"[WARN] skip empty synth frame folder: {frame_dir}")
            continue

        slices = [iio.imread(p) for p in pngs]
        vol = np.stack(slices, axis=0)   # (Z,Y,X)

        if expected_shape is None:
            expected_shape = vol.shape
        elif vol.shape != expected_shape:
            raise RuntimeError(
                f"[ERROR] Synth shape mismatch at frame {frame_dir.name}: "
                f"{vol.shape} != {expected_shape}"
            )

        volumes.append(vol)

    if not volumes:
        raise RuntimeError("[ERROR] No valid synth volumes loaded.")

    data = np.stack(volumes, axis=0)   # (T,Z,Y,X)
    print("[INFO] synth final 4D shape =", data.shape)
    return data

def load_real(folder: Path):
    # 先尝试：real 下面也是按帧子文件夹组织
    frame_dirs = collect_frame_dirs(folder)

    # 如果 real 下存在子文件夹，则优先按“每个子文件夹=1帧”读取
    if frame_dirs:
        print("[INFO] real frame folders found:")
        for p in frame_dirs[:10]:
            print("   ", p.name)
        if len(frame_dirs) > 10:
            print(f"   ... total = {len(frame_dirs)}")

        volumes = []
        expected_shape = None

        for frame_dir in frame_dirs:
            tifs = sorted(
                list(frame_dir.glob("*.tif")) + list(frame_dir.glob("*.tiff")),
                key=lambda p: natural_key(p.name)
            )
            pngs = sorted(list(frame_dir.glob("*.png")), key=lambda p: natural_key(p.name))

            if tifs:
                slices = [tiff.imread(str(p)) for p in tifs]
                # 如果每张 tif 本身就是 2D，则 stack；如果已经是 3D，就直接用第一张
                if slices[0].ndim == 2:
                    vol = np.stack(slices, axis=0)
                elif slices[0].ndim == 3 and len(slices) == 1:
                    vol = slices[0]
                else:
                    raise RuntimeError(f"[ERROR] Unsupported tif structure in {frame_dir}")
            elif pngs:
                slices = [iio.imread(p) for p in pngs]
                vol = np.stack(slices, axis=0)
            else:
                print(f"[WARN] skip empty real frame folder: {frame_dir}")
                continue

            if expected_shape is None:
                expected_shape = vol.shape
            elif vol.shape != expected_shape:
                raise RuntimeError(
                    f"[ERROR] Real shape mismatch at frame {frame_dir.name}: "
                    f"{vol.shape} != {expected_shape}"
                )

            volumes.append(vol)

        if not volumes:
            raise RuntimeError("[ERROR] No valid real volumes loaded from frame folders.")

        data = np.stack(volumes, axis=0)   # (T,Z,Y,X)
        print("[INFO] real final 4D shape =", data.shape)
        return data

    # 否则尝试：real 下面直接是很多 tif 文件
    tif_files = collect_recursive_tifs(folder)
    if tif_files:
        print("[INFO] first 10 real tif files:")
        for p in tif_files[:10]:
            print("   ", p)
        if len(tif_files) > 10:
            print(f"   ... total = {len(tif_files)}")

        vols = []
        expected_shape = None
        for f in tif_files:
            arr = tiff.imread(str(f))
            if arr.ndim != 3:
                raise RuntimeError(f"[ERROR] Real tif is not 3D: {f}, shape={arr.shape}")

            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise RuntimeError(
                    f"[ERROR] Real tif shape mismatch: {f}, {arr.shape} != {expected_shape}"
                )

            vols.append(arr)

        data = np.stack(vols, axis=0)   # (T,Z,Y,X)
        print("[INFO] real final 4D shape =", data.shape)
        return data

    raise RuntimeError(f"[ERROR] No readable real data found under: {folder}")

# =========================================================
# 读取数据
# =========================================================
real = load_real(REAL_DIR)
synth = load_synth_from_png_folders(SYNTH_DIR)

print("[INFO] real shape :", real.shape)
print("[INFO] synth shape:", synth.shape)

# 为了同一个时间轴播放，取共同最小帧数
nT = min(real.shape[0], synth.shape[0])
if real.shape[0] != synth.shape[0]:
    print(f"[WARN] different T: real={real.shape[0]}, synth={synth.shape[0]}, using min T={nT}")

real = real[:nT]
synth = synth[:nT]

# =========================================================
# 打开同一个 napari 窗口
# =========================================================
viewer = napari.Viewer(title="Real + Synth (same napari window)")

layer_real = viewer.add_image(
    real,
    name="real",
    rendering="mip",
    depiction="volume",
    colormap="gray",
)

layer_synth = viewer.add_image(
    synth,
    name="synth",
    rendering="mip",
    depiction="volume",
    colormap="green",
)

# 有些 napari 版本需要单独设 interpolation
try:
    layer_real.interpolation = "linear"
except Exception:
    pass

try:
    layer_synth.interpolation = "linear"
except Exception:
    pass

viewer.dims.axis_labels = ("t", "z", "y", "x")
viewer.dims.ndisplay = 3

# 同一个 napari 窗口里并排显示两个 layer
viewer.grid.enabled = True
viewer.grid.stride = 2

# 不主动改 scale，避免把结构拉坏
# 如果你将来确认 real 的真实体素比例，再单独改 layer_real.scale

# 自动循环播放
fps = 2
interval_ms = int(1000 / fps)

timer = QTimer()

def step():
    t = int(viewer.dims.current_step[0])
    viewer.dims.set_point(0, (t + 1) % nT)

timer.timeout.connect(step)
timer.start(interval_ms)

print(f"[INFO] autoplay started at {fps} fps, total frames = {nT}")

napari.run()
