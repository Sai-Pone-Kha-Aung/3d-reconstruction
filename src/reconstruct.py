"""
Stage 2: Feature Detection, Matching, and Sparse SfM Reconstruction
Extracts high-density SIFT features across single or multi-folder image datasets (e.g. cv-1, cv-2, cv-3, cv-4),
matches image pairs, and runs PyCOLMAP Incremental Structure from Motion (SfM).
"""

import argparse
import shutil
from pathlib import Path
import pycolmap

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(image_dir, folders=None):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = []
    if folders:
        for f in folders:
            sub = image_dir / f
            if not sub.exists():
                print(f"Warning: Subfolder {sub} not found, skipping.")
                continue
            sub_imgs = [p for p in sub.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
            print(f" - Found {len(sub_imgs)} images in {f}")
            images.extend(sub_imgs)
    else:
        # Check if direct images exist or if subfolders exist
        direct_imgs = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if direct_imgs:
            images.extend(direct_imgs)
            print(f" - Found {len(direct_imgs)} images directly in {image_dir.name}")
        
        # Check subfolders
        subdirs = [p for p in image_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        for sub in sorted(subdirs):
            sub_imgs = [p for p in sub.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
            if sub_imgs:
                print(f" - Found {len(sub_imgs)} images in subfolder '{sub.name}'")
                images.extend(sub_imgs)

    return sorted(images)


def run_sfm(
    image_dir,
    output_dir,
    folders=None,
    camera_mode="single",
    matcher="exhaustive",
    max_features=16384,
    peak_threshold=0.006,
    first_octave=-1,
    clean=True,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"

    if clean and database_path.exists():
        print(f"Cleaning previous database at {database_path}...")
        database_path.unlink()
    if clean and sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning images in: {image_dir}")
    images = find_images(image_dir, folders=folders)

    if not images:
        raise FileNotFoundError(f"No valid images found in {image_dir}. Run preprocess.py first.")

    print(f"\nTotal images found for reconstruction: {len(images)}")

    # Camera mode mapping
    mode_map = {
        "single": pycolmap.CameraMode.SINGLE,
        "per_folder": pycolmap.CameraMode.PER_FOLDER,
        "per_image": pycolmap.CameraMode.PER_IMAGE,
    }
    colmap_cam_mode = mode_map.get(camera_mode.lower(), pycolmap.CameraMode.SINGLE)

    # Configure high-density SIFT extraction options
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = max_features
    extraction_options.sift.peak_threshold = peak_threshold
    extraction_options.sift.first_octave = first_octave

    print(f"Step 1/3: Extracting SIFT features (max_features={max_features}, peak_threshold={peak_threshold}, camera_mode={camera_mode})...")
    image_names = [str(p.relative_to(image_dir)) for p in images]
    pycolmap.extract_features(
        database_path,
        image_dir,
        image_names=image_names,
        camera_mode=colmap_cam_mode,
        extraction_options=extraction_options,
    )

    print(f"Step 2/3: Matching features across image pairs (matcher={matcher})...")
    if matcher == "sequential":
        pycolmap.match_sequential(database_path)
    else:
        pycolmap.match_exhaustive(database_path)

    print("Step 3/3: Running Incremental Structure from Motion (SfM)...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)

    if not reconstructions:
        raise RuntimeError(
            "SfM failed to produce a reconstruction. Likely causes: too few overlapping "
            "images, insufficient texture/features on the object, or poor image quality. "
            "Check data/preprocessed for blur warnings and verify photo overlap."
        )

    best_key = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    reconstruction = reconstructions[best_key]

    reconstruction.write(sparse_dir)

    # Export sparse_pc.ply for 3DGS and visualization
    ply_path = output_dir / "sparse_pc.ply"
    reconstruction.export_PLY(ply_path)
    reconstruction.export_PLY(sparse_dir / "sparse_pc.ply")

    print(f"\n==========================================")
    print(f"🎉 Reconstruction complete!")
    print(f"Summary: {reconstruction.summary()}")
    print(f"Registered {reconstruction.num_reg_images()} / {len(images)} images")
    print(f"Sparse point cloud: {len(reconstruction.points3D)} 3D points")
    print(f"Saved reconstruction to: {sparse_dir}")
    print(f"Exported sparse_pc.ply to: {ply_path}")
    print(f"==========================================")

    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sparse SfM reconstruction on single or multi-folder datasets.")
    parser.add_argument(
        "--images", default="data/preprocessed",
        help="Preprocessed image folder (can contain subfolders like cv-1, cv-2, etc.)"
    )
    parser.add_argument(
        "--folders", nargs="+", default=None,
        help="Optional specific subfolders to include (e.g. --folders cv-1 cv-2 cv-3 cv-4)"
    )
    parser.add_argument(
        "--output", default="data/colmap_output/multi_ring_1",
        help="COLMAP output directory"
    )
    parser.add_argument(
        "--camera-mode", choices=["single", "per_folder", "per_image"], default="single",
        help="Camera calibration mode (single = shared camera, per_folder = 1 camera per subfolder)"
    )
    parser.add_argument(
        "--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
        help="Feature matching algorithm (exhaustive matches across all elevation rings)"
    )
    parser.add_argument(
        "--max-features", type=int, default=16384,
        help="Maximum SIFT keypoints per image (default: 16384 for dense textures)"
    )
    parser.add_argument(
        "--peak-threshold", type=float, default=0.006,
        help="SIFT peak threshold (lower detects fainter textures, default: 0.006)"
    )
    parser.add_argument(
        "--no-clean", action="store_false", dest="clean",
        help="Do not clean previous database/sparse folder before running"
    )
    args = parser.parse_args()

    run_sfm(
        image_dir=args.images,
        output_dir=args.output,
        folders=args.folders,
        camera_mode=args.camera_mode,
        matcher=args.matcher,
        max_features=args.max_features,
        peak_threshold=args.peak_threshold,
        clean=args.clean,
    )