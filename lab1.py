"""Lab 1: image enhancement and filtering.

Run: python lab1.py --out results
Your own image: python lab1.py --image images/my_photo.jpg --out my_results
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # also works without a graphical desktop
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parent

def u8(values):
    """Clip and round before conversion to avoid uint8 overflow."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)

def read_rgb(path):
    # imdecode also supports Unicode paths on Windows.
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_image(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"Cannot save: {path}")
    encoded.tofile(str(path))

def gaussian_noise(rgb, sigma=20.0, seed=42):
    """Sigma uses 8-bit intensity units; channels receive independent noise."""
    rng = np.random.default_rng(seed)
    return u8(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape))

def salt_pepper_noise(rgb, amount=0.05, seed=42):
    """An expected fraction amount of pixels is changed, half black and half white."""
    if not 0 <= amount <= 1:
        raise ValueError("amount must be in [0,1]")
    rng = np.random.default_rng(seed)
    mask = rng.random(rgb.shape[:2])
    result = rgb.copy()
    result[mask < amount / 2] = 0
    result[(mask >= amount / 2) & (mask < amount)] = 255
    return result

def filter_bank(rgb):
    """Filters operate on uint8. The settings match the lab parameter table."""
    return {
        "mean5": cv2.blur(rgb, (5, 5), borderType=cv2.BORDER_REFLECT_101),
        "gaussian5": cv2.GaussianBlur(rgb, (5, 5), 1.2,
                                     borderType=cv2.BORDER_REFLECT_101),
        "median5": cv2.medianBlur(rgb, 5),
        "bilateral": cv2.bilateralFilter(rgb, 9, 50, 5,
                                          borderType=cv2.BORDER_REFLECT_101),
        "nlm": nlm_rgb(rgb),
    }

def nlm_rgb(rgb):
    # The Colored variant interprets its input as BGR, so convert explicitly.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    clean = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

def metrics(reference, candidate):
    # Subtract in float to avoid uint8 overflow.
    mse = float(np.mean((reference.astype(np.float64) -
                         candidate.astype(np.float64)) ** 2))
    return {
        "mse": mse,
        "psnr_db": float(peak_signal_noise_ratio(reference, candidate,
                                                  data_range=255)),
        "ssim": float(structural_similarity(reference, candidate,
                         data_range=255, channel_axis=-1)),
    }

def timed_filters(rgb, repeats=3):
    # One warm-up. Time only the filter, excluding saving and metrics.
    funcs = {
        "mean5": lambda: cv2.blur(rgb, (5, 5), borderType=cv2.BORDER_REFLECT_101),
        "gaussian5": lambda: cv2.GaussianBlur(rgb, (5, 5), 1.2,
                                              borderType=cv2.BORDER_REFLECT_101),
        "median5": lambda: cv2.medianBlur(rgb, 5),
        "bilateral": lambda: cv2.bilateralFilter(rgb, 9, 50, 5,
                                             borderType=cv2.BORDER_REFLECT_101),
        "nlm": lambda: nlm_rgb(rgb),
    }
    times = {}
    for name, function in funcs.items():
        function()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            samples.append((time.perf_counter() - start) * 1000)
        times[name] = float(np.median(samples))
    return times

def percentile_stretch(gray, low=2, high=98):
    a, b = np.percentile(gray, [low, high])
    if b <= a:  # constant image: avoid division by zero
        return gray.copy()
    return u8((gray.astype(np.float32) - a) * (255.0 / (b - a)))

def gamma_transform(gray, gamma=0.6):
    """Convention: Iout=255*(Iin/255)**gamma. Gamma<1 brightens midtones."""
    lut = u8(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** gamma)
    return cv2.LUT(gray, lut)

def clahe_rgb(rgb, clip_limit=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2RGB)

def unsharp(rgb, sigma=1.2, amount=1.0):
    src = rgb.astype(np.float32)
    low = cv2.GaussianBlur(src, (0, 0), sigma)
    return u8(src + amount * (src - low))

def panel(images, labels, path, columns=3):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns,
                             figsize=(4 * columns, 3.25 * rows), squeeze=False)
    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  vmin=0, vmax=255)
        ax.set_title(label, fontsize=13)
        ax.axis("off")
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

def run(image=None, out="results", seed=42, sigma=20.0, amount=0.05):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    supplied = ROOT / "images" / "coffee.png"
    rgb = read_rgb(image) if image else (read_rgb(supplied) if supplied.exists()
                                        else data.coffee())
    if min(rgb.shape[:2]) < 7:
        raise ValueError("The image must be at least 7x7 pixels for SSIM.")
    # Use one source and the same noise realization for every method.
    # Resize large custom photographs to at most 1000 pixels in width.
    if rgb.shape[1] > 1000:
        h = round(rgb.shape[0] * 1000 / rgb.shape[1])
        rgb = cv2.resize(rgb, (1000, h), interpolation=cv2.INTER_AREA)
    cv2.setNumThreads(1)
    save_image(out / "clean.png", rgb)
    rows = []
    for kind, noisy in {
        "gaussian": gaussian_noise(rgb, sigma, seed),
        "salt_pepper": salt_pepper_noise(rgb, amount, seed),
    }.items():
        filtered = filter_bank(noisy)
        elapsed = timed_filters(noisy)
        save_image(out / f"{kind}_noisy.png", noisy)
        rows.append({"noise": kind, "method": "noisy", **metrics(rgb, noisy),
                     "time_ms": ""})
        for name, result in filtered.items():
            save_image(out / f"{kind}_{name}.png", result)
            rows.append({"noise": kind, "method": name, **metrics(rgb, result),
                         "time_ms": elapsed[name]})
        panel([rgb, noisy, filtered["mean5"], filtered["gaussian5"],
               filtered["median5"], filtered["bilateral"]],
              ["Original", kind, "Mean 5x5", "Gaussian 5x5, sigma=1.2",
               "Median 5x5", "Bilateral d=9"], out / f"{kind}_comparison.png")
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Controlled simulation of low contrast and uneven illumination.
    base = 70 + 0.45 * gray.astype(np.float32)
    shade = np.linspace(0.45, 1.0, gray.shape[1])[None, :]
    low = u8(base * shade)
    contrast = {
        "gray": gray, "low_contrast": low,
        "stretch": percentile_stretch(low),
        "equalized": cv2.equalizeHist(low),
        "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(low),
        "gamma06": gamma_transform(low, 0.6),
    }
    for name, result in contrast.items():
        save_image(out / f"{name}.png", result)
    panel(list(contrast.values()),
          ["Grayscale", "Low contrast / shadow", "Percentiles 2-98",
           "Global equalization", "CLAHE 2.0, grid 8x8", "Gamma 0.6"],
          out / "contrast_comparison.png")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    histograms = {}
    for ax, (name, result) in zip(axes.flat, contrast.items()):
        counts = np.bincount(result.ravel(), minlength=256)
        histograms[name] = counts.tolist()
        ax.plot(np.arange(256), counts, color="#167d9a", lw=1.3)
        ax.set_title(name)
        ax.set_xlabel("Intensity level")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=150)
    plt.close(fig)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    for name, arr in {"hsv": hsv, "lab": lab, "ycrcb": ycc}.items():
        for i in range(3):
            save_image(out / f"{name}_{i}.png", arr[:, :, i])
    for i, name in enumerate(["r", "g", "b"]):
        plane = np.zeros_like(rgb)
        plane[:, :, i] = rgb[:, :, i]
        save_image(out / f"channel_{name}.png", plane)
    save_image(out / "bgr_shown_as_rgb.png", rgb[:, :, ::-1])
    save_image(out / "color_clahe.png", clahe_rgb(rgb))
    bad = np.stack([cv2.equalizeHist(rgb[:, :, i]) for i in range(3)], axis=-1)
    save_image(out / "color_equalized_each_channel.png", bad)
    save_image(out / "unsharp_clean.png", unsharp(rgb))
    noisy = gaussian_noise(rgb, sigma, seed)
    save_image(out / "unsharp_noisy.png", unsharp(noisy))
    save_image(out / "unsharp_denoised.png", unsharp(filter_bank(noisy)["bilateral"]))
    # Ideal CMYK for learning the formula. This model does not simulate a printer.
    normal = rgb.astype(np.float32) / 255.0
    k = 1 - normal.max(axis=-1)
    denominator = 1 - k
    cmy = np.zeros_like(normal)
    np.divide(1 - normal - k[:, :, None], denominator[:, :, None],
              out=cmy, where=denominator[:, :, None] > 0)
    for arr, name in zip([cmy[:, :, 0], cmy[:, :, 1], cmy[:, :, 2], k],
                         ["c", "m", "y", "k"]):
        save_image(out / f"cmyk_{name}.png", u8(arr * 255))

    # Window-size effect: use the same noisy data for every k.
    sweep = []
    sp = salt_pepper_noise(rgb, amount, seed)
    for size in [3, 5, 7, 9]:
        med = cv2.medianBlur(sp, size)
        save_image(out / f"sweep_median{size}.png", med)
        sweep.append({"kernel": size, **metrics(rgb, med)})
    with open(out / "summary.json", "w", encoding="utf-8") as file:
        json.dump({"parameters": {"seed": seed, "sigma": sigma,
                    "salt_pepper_amount": amount, "shape": list(rgb.shape)},
                   "versions": {"python": platform.python_version(),
                    "opencv": cv2.__version__, "numpy": np.__version__,
                    "scikit_image": skimage.__version__,
                    "matplotlib": matplotlib.__version__},
                   "metrics": rows, "median_sweep": sweep,
                   "histograms": histograms}, file, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out.resolve()}")
    for kind in ["gaussian", "salt_pepper"]:
        winner = max((r for r in rows if r["noise"] == kind and r["method"] != "noisy"),
                     key=lambda r: r["ssim"])
        print(f"{kind}: highest SSIM = {winner['method']} ({winner['ssim']:.4f})")
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--amount", type=float, default=0.05)
    args = parser.parse_args()
    if args.sigma < 0:
        parser.error("--sigma must be non-negative")
    run(args.image, args.out, args.seed, args.sigma, args.amount)
