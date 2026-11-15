#!/usr/bin/env python3
"""
Helper script to package images and COLMAP sparse reconstruction into
a standardized dataset zip archive for 3D Gaussian Splatting (3DGS) and NeRF
on Google Colab or Kaggle.
"""

import os
import shutil
from pathlib import Path

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package COLMAP reconstruction and images into a neural dataset archive (3DGS / NeRF / NeuS)."
    )
    parser.add_argument(
        "--colmap-dir",
        default=None,
        help="COLMAP output directory (e.g. data/colmap_output/multi_ring_garuda)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Preprocessed images folder (e.g. data/preprocessed_fuyee or data/preprocessed_png)",
    )
    parser.add_argument(
        "--output",
        default="dataset_neural",
        help="Output directory and zip name (default: dataset_neural)",
    )
    return parser.parse_args()


def find_sparse_dir(colmap_dir, root):
    if colmap_dir:
        candidates = [
            colmap_dir / "sparse" / "0",
            colmap_dir / "sparse" / "1",
            colmap_dir / "sparse",
            colmap_dir,
        ]
    else:
        candidates = [
            root / "data" / "colmap_output" / "multi_ring_garuda" / "sparse",
            root / "data" / "colmap_output" / "multi_ring_png" / "sparse" / "1",
            root / "data" / "colmap_output" / "multi_ring_png" / "sparse",
            root / "data" / "colmap_output" / "multi_ring_png" / "sparse" / "0",
            root / "data" / "colmap_output" / "multi_ring_1" / "sparse" / "1",
            root / "data" / "colmap_output" / "multi_ring_1" / "sparse",
            root / "data" / "colmap_output" / "multi_ring" / "sparse" / "1",
            root / "data" / "colmap_output" / "multi_ring" / "sparse",
            root / "data" / "colmap_output" / "multi_ring" / "sparse" / "0",
        ]

    for p in candidates:
        if p.exists() and ((p / "images.bin").exists() or (p / "images.txt").exists()):
            return p
    return None


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    colmap_dir = Path(args.colmap_dir).resolve() if args.colmap_dir else None
    images_dir_arg = Path(args.images_dir).resolve() if args.images_dir else None

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    dataset_images_dir = out_dir / "images"
    dataset_sparse_dir = out_dir / "sparse" / "0"

    # Clean up previous dataset folder
    if out_dir.exists():
        shutil.rmtree(out_dir)

    dataset_images_dir.mkdir(parents=True, exist_ok=True)
    dataset_sparse_dir.mkdir(parents=True, exist_ok=True)

    print("📦 1. Locating COLMAP sparse reconstruction...")
    src_sparse = find_sparse_dir(colmap_dir, root)

    rec = None
    if src_sparse:
        print(f"  Found sparse model in: {src_sparse}")
        for f in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
            src_file = src_sparse / f
            if src_file.exists():
                shutil.copy2(src_file, dataset_sparse_dir / f)
                print(f"  - Copied {f}")

        try:
            import pycolmap
            rec = pycolmap.Reconstruction(dataset_sparse_dir)
        except Exception as e:
            print(f"  Note: pycolmap reconstruction load notice: {e}")
    else:
        print("  ❌ Could not find sparse model files (.bin or .txt)!")

    # Locate source images directory
    candidate_img_dirs = []
    if images_dir_arg:
        candidate_img_dirs.append(images_dir_arg)
    candidate_img_dirs.extend([
        root / "data" / "preprocessed_fuyee",
        root / "data" / "preprocessed_png",
        root / "data" / "preprocessed",
    ])

    resolved_images_dir = None
    if rec is not None and len(rec.images) > 0:
        sample_img_name = list(rec.images.values())[0].name
        for cand in candidate_img_dirs:
            if (cand / sample_img_name).exists():
                resolved_images_dir = cand
                break

    if resolved_images_dir is None:
        for cand in candidate_img_dirs:
            if cand.exists():
                resolved_images_dir = cand
                break

    print(f"\n📦 2. Collecting images from: {resolved_images_dir}...")
    copied_images = 0
    subfolders = set()

    if rec is not None and resolved_images_dir is not None:
        # Copy registered images directly
        for img in rec.images.values():
            src_img = resolved_images_dir / img.name
            dst_img = dataset_images_dir / img.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_images += 1
                if "/" in img.name or "\\" in img.name:
                    subfolders.add(Path(img.name).parent.as_posix())
            else:
                print(f"  ⚠️ Warning: Image not found at {src_img}")
    elif resolved_images_dir is not None and resolved_images_dir.exists():
        # Fallback copy all image files
        for img_path in sorted(list(resolved_images_dir.rglob("*.png")) + list(resolved_images_dir.rglob("*.jpg"))):
            rel_path = img_path.relative_to(resolved_images_dir)
            dst_img = dataset_images_dir / rel_path
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)
            copied_images += 1
            if len(rel_path.parts) > 1:
                subfolders.add(rel_path.parts[0])

    # Create convenient symlinks for subfolders at root level
    for sub in subfolders:
        root_symlink = out_dir / sub
        if not root_symlink.exists():
            try:
                os.symlink(f"images/{sub}", root_symlink)
            except OSError:
                pass

    print(f"  ✅ Collected {copied_images} images into {dataset_images_dir.name}")

    # Generate transforms.json & meta_data.json using pycolmap
    print("\n📦 3. Generating transforms.json & meta_data.json (for NeuS / Nerfstudio)...")
    try:
        import numpy as np

        if rec is None:
            import pycolmap
            rec = pycolmap.Reconstruction(dataset_sparse_dir)

        cam = list(rec.cameras.values())[0]

        w = cam.width
        h = cam.height
        fl_x = float(cam.params[0])
        fl_y = float(cam.params[0])
        cx = float(cam.params[1])
        cy = float(cam.params[2])

        K = [
            [fl_x, 0.0, cx],
            [0.0, fl_y, cy],
            [0.0, 0.0, 1.0]
        ]

        frames_nerf = []
        sdf_frames = []

        for img_id, img in rec.images.items():
            rigid = img.cam_from_world()
            c2w_colmap = np.eye(4)
            c2w_colmap[:3, :] = rigid.inverse().matrix()

            # OpenGL convention for NeRF: flip Y and Z
            c2w_nerf = c2w_colmap.copy()
            c2w_nerf[0:3, 1:3] *= -1

            rel_path = f"images/{img.name}"

            frames_nerf.append({
                "file_path": rel_path,
                "transform_matrix": c2w_nerf.tolist()
            })

            sdf_frames.append({
                "rgb_path": rel_path,
                "camtoworld": c2w_colmap.tolist(),
                "intrinsics": K
            })

        # Export COLMAP points to PLY so splatfacto can initialize Gaussians from real 3D points
        ply_path = out_dir / "sparse_pc.ply"
        rec.export_PLY(ply_path)
        print("  - Exported sparse_pc.ply for 3DGS initialization")

        transforms_dict = {
            "w": w,
            "h": h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "camera_model": "OPENCV",
            "ply_file_path": "sparse_pc.ply",
            "frames": frames_nerf
        }

        meta_dict = {
            "camera_model": "OPENCV",
            "height": h,
            "width": w,
            "has_sensor_depth": False,
            "has_monocular_depth": False,
            "has_monocular_normal": False,
            "has_foreground_mask": False,
            "pairs": None,
            "world_size": 2.0,
            "scene_box": {
                "aabb": [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
                "near": 0.05,
                "far": 3.5,
                "radius": 1.5,
                "collider_type": "box"
            },
            "frames": sdf_frames
        }

        with open(out_dir / "transforms.json", "w") as f:
            json.dump(transforms_dict, f, indent=2)
        print("  - Created transforms.json")

        with open(out_dir / "meta_data.json", "w") as f:
            json.dump(meta_dict, f, indent=2)
        print("  - Created meta_data.json (fixes NeuS export error!)")

    except Exception as e:
        print(f"  ⚠️ Warning: Could not generate transforms.json/meta_data.json: {e}")

    # Create zip archive
    zip_target = out_dir
    print(f"\n📦 4. Compressing into {zip_target.name}.zip...")
    archive_created = shutil.make_archive(str(zip_target), "zip", out_dir)
    size_mb = os.path.getsize(archive_created) / (1024 * 1024)
    print(f"\n🎉 SUCCESS: Created '{Path(archive_created).name}' ({size_mb:.1f} MB)")
    print("Ready for 3DGS / NeuS training on Kaggle or Colab.")


if __name__ == "__main__":
    main()
