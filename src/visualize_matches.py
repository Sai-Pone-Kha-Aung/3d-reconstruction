"""
Feature Matching & Keypoint Visualizer for Reports / DOCX
Exports high-resolution (300 DPI) SIFT keypoint detection and two-view match images
directly from COLMAP database.db across single or multi-folder datasets (cv-1, cv-2, cv-3, cv-4).
"""

import argparse
import sqlite3
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


def read_keypoints(cursor, image_id):
    cursor.execute("SELECT data, rows, cols FROM keypoints WHERE image_id = ?", (image_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    data, rows, cols = row
    if data is None or rows == 0:
        return None
    kps = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
    return kps


def find_image_file(images_dir, name):
    images_dir = Path(images_dir)
    # 1. Direct path
    p = images_dir / name
    if p.exists():
        return p
    # 2. In subfolders
    matches = list(images_dir.rglob(Path(name).name))
    if matches:
        return matches[0]
    # 3. Check data/frames fallback
    frames_dir = Path("data/frames")
    if frames_dir.exists():
        p_frame = frames_dir / name
        if p_frame.exists():
            return p_frame
        frame_matches = list(frames_dir.rglob(Path(name).name))
        if frame_matches:
            return frame_matches[0]
    return None


def export_colmap_matches(
    db_path="data/colmap_output/multi_ring_png/database.db",
    images_dir="data/preprocessed_png",
    output_path="outputs/feature_matches_png.png",
    max_matches_draw=150,
    pair_rank=0,
    cross_folder_only=False,
):
    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    if not db_path.exists():
        # Fallback to source1 database if multi_ring does not exist yet
        fallback_db = Path("data/colmap_output/multi_ring_1/database.db")
        if fallback_db.exists():
            print(f"Note: {db_path} not found. Falling back to {fallback_db}")
            db_path = fallback_db
        else:
            raise FileNotFoundError(f"COLMAP database not found at {db_path}. Run reconstruct.py first.")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT pair_id, rows, data FROM two_view_geometries WHERE rows > 20 ORDER BY rows DESC LIMIT 100;"
    )
    all_pairs = cursor.fetchall()

    if not all_pairs:
        print("No two-view geometries found in database.")
        return

    # Filter for cross-folder or ranked pairs
    filtered_pairs = []
    for p_id, n_inliers, m_data in all_pairs:
        id1, id2 = pair_id_to_image_ids(p_id)
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (id1,))
        n1 = cursor.fetchone()[0]
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (id2,))
        n2 = cursor.fetchone()[0]

        is_cross = Path(n1).parent != Path(n2).parent
        if cross_folder_only and not is_cross:
            continue
        filtered_pairs.append((p_id, n_inliers, m_data, id1, id2, n1, n2))

    if not filtered_pairs:
        filtered_pairs = [
            (p[0], p[1], p[2], *pair_id_to_image_ids(p[0]), "", "")
            for p in all_pairs[:10]
        ]

    pair_rank = min(pair_rank, len(filtered_pairs) - 1)
    pair_id, num_inliers, match_data, id1, id2, name1, name2 = filtered_pairs[pair_rank]

    if not name1:
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (id1,))
        name1 = cursor.fetchone()[0]
        cursor.execute("SELECT name FROM images WHERE image_id = ?", (id2,))
        name2 = cursor.fetchone()[0]

    kps1 = read_keypoints(cursor, id1)
    kps2 = read_keypoints(cursor, id2)

    matches_arr = np.frombuffer(match_data, dtype=np.uint32).reshape((num_inliers, 2))

    p1 = find_image_file(images_dir, name1)
    p2 = find_image_file(images_dir, name2)

    if p1 is None or p2 is None:
        raise FileNotFoundError(f"Could not find image files for {name1} or {name2} in {images_dir}")

    img1 = cv2.imread(str(p1))
    img2 = cv2.imread(str(p2))

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    out_h = max(h1, h2)
    out_w = w1 + w2

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1 : w1 + w2] = img2_rgb

    # Sample matches for clean visualization
    sample_step = max(1, len(matches_arr) // max_matches_draw)
    sampled_matches = matches_arr[::sample_step]

    # Create figure with high DPI for DOCX reports
    plt.figure(figsize=(14, 7), dpi=300)
    plt.imshow(canvas)

    # Plot lines and points with a vibrant colormap
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(sampled_matches)))

    for idx, (m1, m2) in enumerate(sampled_matches):
        pt1 = kps1[m1, :2]
        pt2 = kps2[m2, :2]
        c = colors[idx]

        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0] + w1, pt2[1]

        plt.plot([x1, x2], [y1, y2], color=c, alpha=0.75, linewidth=1.2)
        plt.scatter([x1, x2], [y1, y2], color=c, s=12, edgecolor="white", linewidth=0.5, zorder=3)

    plt.title(
        f"SIFT Two-View Inlier Feature Matching ({name1} ⟷ {name2})\n"
        f"Total Inlier Geometric Matches: {num_inliers} (Displaying {len(sampled_matches)} Sampled Correspondences)",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Exported COLMAP feature matches to: {output_path}")
    print(f"Matched pair: {name1} <-> {name2} with {num_inliers} inlier points.")
    return output_path


def export_sift_keypoints_visual(
    image_path=None,
    images_dir="data/preprocessed",
    output_path="outputs/sift_keypoints.png",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path is None:
        # Find first available preprocessed image (.png or .jpg)
        available = sorted(
            list(Path(images_dir).rglob("*.png")) + list(Path(images_dir).rglob("*.jpg"))
        )
        if not available:
            print(f"No images found in {images_dir}")
            return
        image_path = available[len(available) // 2]  # pick middle image

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)

    img_kp = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )

    cv2.imwrite(str(output_path), img_kp)
    print(f"Exported SIFT keypoints visualization ({len(keypoints)} detected keypoints from {image_path.name}) to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export feature matching images for reports / DOCX.")
    parser.add_argument(
        "--db", default="data/colmap_output/multi_ring_png/database.db",
        help="Path to COLMAP database.db (e.g. data/colmap_output/multi_ring/database.db or data/colmap_output/source1/database.db)"
    )
    parser.add_argument(
        "--images", default="data/preprocessed_png",
        help="Path to preprocessed images directory containing cv-1, cv-2, etc."
    )
    parser.add_argument(
        "--output", default="outputs/feature_matches_colmap_png.png",
        help="Output match image path (.png)"
    )
    parser.add_argument(
        "--max-matches", type=int, default=150,
        help="Max correspondences to draw"
    )
    parser.add_argument(
        "--pair-rank", type=int, default=0,
        help="Rank of matched pair to export (0 = top best match, 1 = 2nd best, etc.)"
    )
    parser.add_argument(
        "--cross-folder", action="store_true",
        help="Force selecting a match between two different elevation folders (e.g. cv-1 <-> cv-2)"
    )
    args = parser.parse_args()

    # 1. Export COLMAP geometric inlier match image (300 DPI)
    export_colmap_matches(
        db_path=args.db,
        images_dir=args.images,
        output_path=args.output,
        max_matches_draw=args.max_matches,
        pair_rank=args.pair_rank,
        cross_folder_only=args.cross_folder,
    )

    # 2. Export single-image SIFT keypoint detection
    export_sift_keypoints_visual(
        images_dir=args.images,
        output_path="outputs/sift_keypoints_1.png",
    )
