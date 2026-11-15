"""
Stage 1 (Variant): Image Preprocessing & Background Removal to Transparent PNG
Preprocesses images for photogrammetry / neural rendering:
- Supports processing a single image file or an entire directory
- Resizes to standard width while maintaining aspect ratio
- Performs blur assessment using Laplacian variance
- Removes background using rembg (isnet-general-use, u2net, or u2netp)
- Exports cleanly as RGBA transparent PNG (no black background composite)
"""

import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def blur_score(gray_img):
    """Computes Laplacian variance to assess image sharpness/blur."""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def process_single_image(
    img_path: Path,
    out_path: Path,
    session,
    target_width: int = 1280,
    blur_threshold: float = 80.0,
    remove_bg: bool = True,
    from_black_bg: bool = False,
    black_threshold: int = 10,
):
    """
    Processes a single image:
    1. Reads image
    2. Resizes while preserving aspect ratio
    3. Calculates blur score
    4. Removes background to transparent RGBA (either via rembg or black-thresholding)
    5. Saves as PNG
    """
    img_path = Path(img_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Could not read {img_path}")
        return None

    # 2. Resize maintaining aspect ratio
    h, w = img.shape[:2]
    if target_width and w != target_width:
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    # 3. Blur check
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    score = blur_score(gray)
    is_blurry = score < blur_threshold

    # 4. Background removal
    if remove_bg:
        if from_black_bg:
            # For images already composited onto solid black:
            # Convert black (or near black <= black_threshold) to transparent
            b, g, r = cv2.split(img_resized)
            # Alpha is 0 where all RGB channels are below threshold
            mask = ((b > black_threshold) | (g > black_threshold) | (r > black_threshold)).astype(np.uint8) * 255
            # Soften mask edges slightly
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            rgba = cv2.merge([b, g, r, mask])
            rgba_pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        elif session is not None:
            from rembg import remove
            # Convert BGR (cv2) to RGB (PIL)
            pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            rgba_pil = remove(pil_img, session=session)
        else:
            # Convert to RGBA with full opacity
            rgba_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA))
    else:
        # No BG removal, but output RGBA PNG
        rgba_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA))

    # 5. Save as PNG
    out_path_png = out_path.with_suffix(".png")
    rgba_pil.save(out_path_png, format="PNG")

    return {
        "input": str(img_path),
        "output": str(out_path_png),
        "blur_score": round(score, 1),
        "is_blurry": is_blurry,
        "resolution": rgba_pil.size,
    }


def preprocess_to_png(
    input_path: str,
    output_path: str,
    target_width: int = 1280,
    blur_threshold: float = 80.0,
    remove_bg: bool = True,
    model_name: str = "birefnet-general-lite",
    from_black_bg: bool = False,
    clean: bool = False,
):
    inp = Path(input_path)
    out = Path(output_path)

    # Initialize rembg session if needed
    session = None
    if remove_bg and not from_black_bg:
        try:
            from rembg import new_session
            print(f"🤖 Initializing rembg session (model={model_name})...")
            session = new_session(model_name)
        except ImportError:
            print("⚠️ Warning: rembg is not installed. Background removal will be skipped.")
            remove_bg = False

    # Single file mode
    if inp.is_file():
        if out.is_dir() or out.suffix == "":
            out_file = out / f"{inp.stem}.png"
        else:
            out_file = out.with_suffix(".png")

        print(f"🎯 Processing single image: {inp} -> {out_file}")
        res = process_single_image(
            img_path=inp,
            out_path=out_file,
            session=session,
            target_width=target_width,
            blur_threshold=blur_threshold,
            remove_bg=remove_bg,
            from_black_bg=from_black_bg,
        )
        if res:
            print(f"✅ Success: Saved {res['output']} (size={res['resolution']}, blur_score={res['blur_score']})")
        return [res] if res else []

    # Directory mode
    if not inp.is_dir():
        print(f"❌ Input path {inp} does not exist!")
        return []

    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in inp.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        print(f"❌ No images found in {inp}")
        return []

    print(f"📁 Preprocessing {len(image_paths)} images from {inp} -> {out} (remove_bg={remove_bg}, model={model_name})...")

    results = []
    flagged_blur = []

    for idx, img_path in enumerate(image_paths):
        out_file = out / f"{img_path.stem}.png"
        res = process_single_image(
            img_path=img_path,
            out_path=out_file,
            session=session,
            target_width=target_width,
            blur_threshold=blur_threshold,
            remove_bg=remove_bg,
            from_black_bg=from_black_bg,
        )
        if res:
            results.append(res)
            if res["is_blurry"]:
                flagged_blur.append((img_path.name, res["blur_score"]))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f" [{idx + 1}/{len(image_paths)}] Processed {img_path.name} -> {out_file.name}", flush=True)

    print(f"\n🎉 Done! Successfully preprocessed {len(results)} images as transparent PNGs -> {out}")
    if flagged_blur:
        print(f"\n⚠️ Flagged {len(flagged_blur)} potentially blurry images (score < {blur_threshold}):")
        for name, score in flagged_blur[:10]:
            print(f" - {name} (sharpness: {score})")
        if len(flagged_blur) > 10:
            print(f" ... and {len(flagged_blur) - 10} more.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images to transparent PNG with background removal.")
    parser.add_argument(
        "--input",
        default="data/frames/cv-4/frame_0000.jpg",
        help="Path to an image file OR folder of images (default: data/frames/cv-4/frame_0000.jpg)",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed_png_test/frame_0000.png",
        help="Path to output file OR output folder (default: data/preprocessed_png_test/frame_0000.png)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Target image width (0 to keep original)")
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="Laplacian variance blur threshold")
    parser.add_argument(
        "--model",
        default="birefnet-general-lite",
        help="rembg model name (e.g. birefnet-general-lite, birefnet-general, isnet-general-use, u2net, u2netp)",
    )
    parser.add_argument("--no-remove-bg", action="store_false", dest="remove_bg", help="Disable background removal")
    parser.add_argument(  
        "--from-black-bg",
        action="store_true",
        help="Convert existing black background directly to transparent instead of re-running rembg neural net",
    )
    parser.add_argument("--clean", action="store_true", help="Clean destination directory before running")

    args = parser.parse_args()

    preprocess_to_png(
        input_path=args.input,
        output_path=args.output,
        target_width=args.width,
        blur_threshold=args.blur_threshold,
        remove_bg=args.remove_bg,
        model_name=args.model,
        from_black_bg=args.from_black_bg,
        clean=args.clean,
    )
