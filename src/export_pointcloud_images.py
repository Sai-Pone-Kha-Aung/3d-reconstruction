"""
Point Cloud Image Exporter for Presentation Slides & Reports
Extracts and renders high-definition (1080p/4K) multi-angle images from COLMAP sparse_pc.ply
and visualizes registered camera poses across the 4 elevation rings (cv-1, cv-2, cv-3, cv-4).

Usage:
    python src/export_pointcloud_images.py
    python src/export_pointcloud_images.py --interactive
    python src/export_pointcloud_images.py --theme white
"""

import argparse
import os
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# Ring elevation colors (matching modern presentation theme)
RING_COLORS = {
    'cv-1': [0.06, 0.75, 0.95],  # Cyan / Bright Blue (Top ring)
    'cv-2': [0.45, 0.40, 0.95],  # Indigo / Violet (High oblique)
    'cv-3': [0.10, 0.85, 0.55],  # Emerald Green (Eye level)
    'cv-4': [0.98, 0.60, 0.10],  # Amber / Orange (Underside pass)
    'default': [0.90, 0.90, 0.90]
}


def create_camera_frustum(cam_center, R_world_to_cam, scale=0.08, color=[1.0, 0.8, 0.2]):
    """Creates a 3D pyramid (LineSet) representing camera pose and viewing direction."""
    # Camera coordinate system: +Z is forward, +X is right, +Y is down
    R_cam_to_world = R_world_to_cam.T
    
    # Frustum pyramid apex at camera center
    apex = cam_center
    
    # 4 base corners in camera coordinates
    w, h, d = 0.5 * scale, 0.3 * scale, scale
    corners_cam = np.array([
        [-w, -h, d],
        [ w, -h, d],
        [ w,  h, d],
        [-w,  h, d],
    ])
    
    # Transform base corners to world coordinates
    corners_world = (R_cam_to_world @ corners_cam.T).T + cam_center
    
    points = np.vstack([apex, corners_world])
    
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Apex to base rays
        [1, 2], [2, 3], [3, 4], [4, 1]   # Base rectangle
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def render_view(geometries, center, front, up, zoom=0.75, width=1920, height=1080, bg_color=[0.05, 0.07, 0.12], point_size=4.5):
    """Offscreen render a single high-definition view with Open3D."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    for g in geometries:
        vis.add_geometry(g)
        
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.point_size = float(point_size)
    opt.light_on = False
    
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)
    
    vis.poll_events()
    vis.update_renderer()
    
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    img_np = (np.asarray(image) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_collage(image_paths, output_path, title="Sparse SfM Point Cloud (53k Points) & 274 Camera Poses"):
    """Combines 4 renders into a sleek 2x2 presentation slide panel with subtitles."""
    imgs = [Image.open(p) for p in image_paths]
    w, h = imgs[0].size
    thumb_w, thumb_h = 960, 540
    
    resized = [im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS) for im in imgs]
    
    header_h = 80
    collage = Image.new("RGB", (thumb_w * 2, thumb_h * 2 + header_h), (11, 15, 25))
    draw = ImageDraw.Draw(collage)
    
    # Paste images
    collage.paste(resized[0], (0, header_h))
    collage.paste(resized[1], (thumb_w, header_h))
    collage.paste(resized[2], (0, header_h + thumb_h))
    collage.paste(resized[3], (thumb_w, header_h + thumb_h))
    
    # Text headers
    subtitles = [
        "A) 3D Isometric Overview",
        "B) Top-Down Radial Symmetry",
        "C) Side Elevation & Vertical Profile",
        "D) 4-Elevation Camera Trajectories (274 Poses)"
    ]
    
    # Subtitle bars
    for idx, (x, y) in enumerate([
        (0, header_h),
        (thumb_w, header_h),
        (0, header_h + thumb_h),
        (thumb_w, header_h + thumb_h)
    ]):
        draw.rectangle([x + 15, y + 15, x + 380, y + 50], fill=(23, 32, 54, 200))
        draw.text((x + 25, y + 22), subtitles[idx], fill=(255, 255, 255))
        
    # Main Header Bar
    draw.rectangle([0, 0, thumb_w * 2, header_h], fill=(16, 23, 39))
    draw.text((30, 24), title, fill=(248, 250, 252))
    
    collage.save(output_path, quality=95)
    print(f"🎉 Created presentation collage: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export HD point cloud images from sparse_pc.ply")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/colmap_output/multi_ring_png/sparse_pc.ply"))
    parser.add_argument("--sparse-dir", "-s", type=Path, default=Path("data/colmap_output/multi_ring_png/sparse"))
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("outputs/renders_sparse"))
    parser.add_argument("--theme", choices=["dark", "white"], default="dark", help="Background theme")
    parser.add_argument("--point-size", type=float, default=4.5, help="Point display size")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive 3D viewer")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading point cloud from: {args.input}")
    pcd = o3d.io.read_point_cloud(str(args.input))
    print(f"Total points: {len(pcd.points):,}")
    
    # Filter statistical outliers to tightly center object
    print("Filtering outlier noise for tight framing...")
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)
    print(f"Filtered points: {len(clean_pcd.points):,}")
    
    bbox = clean_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    print(f"Point cloud center: {center.round(3)}, extent: {bbox.get_extent().round(3)}")
    
    if args.interactive:
        print("\nOpening interactive 3D viewer...")
        print("Controls: Mouse to rotate/pan/zoom. Press [H] for help, [S] to capture screen, [Q] or [ESC] to exit.")
        o3d.visualization.draw_geometries(
            [clean_pcd],
            window_name="Thai Parasol - Sparse Point Cloud Inspector",
            width=1400,
            height=900
        )
        return

    # Theme colors
    bg_color = [0.04, 0.06, 0.10] if args.theme == "dark" else [0.98, 0.98, 0.98]
    
    # 1. Load Camera Frustums if COLMAP reconstruction is available
    camera_frustums = []
    if HAS_PYCOLMAP and args.sparse_dir.exists():
        print(f"Loading camera poses from: {args.sparse_dir}...")
        try:
            rec = pycolmap.Reconstruction(str(args.sparse_dir))
            print(f"Loaded {len(rec.images)} registered cameras")
            
            # Determine frustum scale proportional to scene size
            extent_max = max(bbox.get_extent())
            frustum_scale = extent_max * 0.055
            
            for img_id, img in rec.images.items():
                cam_center = img.projection_center()
                R_w2c = img.cam_from_world().rotation.matrix()
                
                # Identify pass folder
                color = RING_COLORS['default']
                for k in ['cv-1', 'cv-2', 'cv-3', 'cv-4']:
                    if k in img.name:
                        color = RING_COLORS[k]
                        break
                        
                frustum = create_camera_frustum(cam_center, R_w2c, scale=frustum_scale, color=color)
                camera_frustums.append(frustum)
        except Exception as e:
            print(f"Warning: Could not load cameras: {e}")

    # View Configurations: (name, front, up, zoom, include_cameras, point_size)
    views = [
        ("01_sparse_isometric_3d.png", [0.6, -0.8, 0.35], [0.0, 0.0, 1.0], 0.72, False, args.point_size),
        ("02_sparse_top_down.png", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 0.70, False, args.point_size),
        ("03_sparse_side_elevation.png", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.72, False, args.point_size),
        ("04_sparse_underside.png", [0.4, 0.5, -0.8], [0.0, 0.0, 1.0], 0.75, False, args.point_size + 0.5),
        ("05_sparse_with_cameras.png", [0.7, -0.7, 0.45], [0.0, 0.0, 1.0], 0.45, True, 3.5),
    ]

    saved_images = []
    print("\nRendering high-definition presentation images...")
    for filename, front, up, zoom, with_cams, pt_sz in views:
        out_file = args.output_dir / filename
        geoms = [clean_pcd] + (camera_frustums if with_cams else [])
        
        img = render_view(
            geoms,
            center=center,
            front=front,
            up=up,
            zoom=zoom,
            bg_color=bg_color,
            point_size=pt_sz
        )
        img.save(out_file, quality=95)
        print(f"✅ Saved: {out_file}")
        saved_images.append(out_file)

    # Create 2x2 presentation panel
    collage_path = args.output_dir / "00_sparse_multiview_panel.png"
    selected_for_collage = [saved_images[0], saved_images[1], saved_images[2], saved_images[4]]
    create_collage(selected_for_collage, collage_path)
    
    print("\n=======================================================")
    print(f"🎉 All point cloud images successfully exported to: {args.output_dir}/")
    print("=======================================================")


if __name__ == "__main__":
    main()
