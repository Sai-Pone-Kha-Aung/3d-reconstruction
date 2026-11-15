"""
Stage 1: Image Preprocessing & Background Removal
Preprocesses images for COLMAP photogrammetry:
- Resizes to standard width while maintaining aspect ratio
- Performs blur assessment using Laplacian variance
- Removes background using fast rembg model (u2netp) and composites onto solid black background
"""

import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def blur_score(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def preprocess_images(
    input_dir,
    output_dir,
    target_width=1280,
    blur_threshold=80.0,
    remove_bg=True,
    model_name="u2netp",
    bg_color=(0, 0, 0),
    clean=True,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    session = None
    if remove_bg:
        try:
            from rembg import new_session, remove
            print(f"Initializing rembg background removal session (model={model_name})...")
            session = new_session(model_name)
        except ImportError:
            print("Warning: rembg is not installed. Background removal will be skipped.")
            remove_bg = False

    flagged = []
    kept = 0

    print(f"Preprocessing {len(image_paths)} images from {input_dir} (remove_bg={remove_bg})...")

    for idx, img_path in enumerate(image_paths):
        # 1. Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f" [skip] Could not read {img_path.name}")
            continue

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
        if score < blur_threshold:
            flagged.append((img_path.name, round(score, 1)))

        # 4. Background removal (if enabled)
        if remove_bg and session is not None:
            # Convert BGR (cv2) to RGB (PIL)
            pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            rgba_img = remove(pil_img, session=session)

            # Create solid background (black prevents SIFT keypoint detection)
            bg = Image.new("RGBA", rgba_img.size, (*bg_color, 255))
            composite = Image.alpha_composite(bg, rgba_img).convert("RGB")

            # Convert back to cv2 BGR format
            final_img = cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR)
        else:
            final_img = img_resized

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), final_img)
        kept += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f" [{idx + 1}/{len(image_paths)}] Processed {img_path.name}", flush=True)

    print(f"\nDone! Successfully processed {kept} images -> {output_dir}")
    if flagged:
        print(f"\nFlagged {len(flagged)} potentially blurry images (Laplacian variance < {blur_threshold}):")
        for name, score in flagged[:10]:
            print(f" - {name} (sharpness: {score})")
        if len(flagged) > 10:
            print(f" ... and {len(flagged) - 10} more.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for 3D reconstruction.")
    parser.add_argument("--input", default="data/frames/cv-4", help="Folder of raw frames/photos")
    parser.add_argument("--output", default="data/preprocessed/cv-4", help="Folder of preprocessed photos")
    parser.add_argument("--width", type=int, default=1280, help="Target image width (preserve aspect ratio)")
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="Laplacian variance blur threshold")
    parser.add_argument("--model", default="u2netp", help="rembg model name (u2netp, silueta, u2net)")
    parser.add_argument("--remove-bg", action="store_true", default=True, help="Remove background with rembg")
    parser.add_argument("--no-remove-bg", action="store_false", dest="remove_bg", help="Disable background removal")
    args = parser.parse_args()

    preprocess_images(
        input_dir=args.input,
        output_dir=args.output,
        target_width=args.width,
        blur_threshold=args.blur_threshold,
        remove_bg=args.remove_bg,
        model_name=args.model,
    )