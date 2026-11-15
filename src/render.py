"""
Stage 4: 3D Visualization & Studio Rendering
Loads the reconstructed 3D mesh or point cloud and provides:
- Interactive Open3D 3D viewer (view mode)
- Studio-grade 360° turntable multi-angle snapshot rendering with custom lighting and antialiasing
"""

import argparse
from pathlib import Path
import open3d as o3d


def load_geometry(geometry_path):
    geometry_path = Path(geometry_path)
    if not geometry_path.exists():
        # Fallback search in outputs folder
        if geometry_path.name == "multi_ring_mesh.ply" and (Path("outputs/parasol_mesh.ply")).exists():
            geometry_path = Path("outputs/parasol_mesh.ply")
        elif geometry_path.name == "parasol_mesh.ply" and (Path("outputs/multi_ring_mesh.ply")).exists():
            geometry_path = Path("outputs/multi_ring_mesh.ply")
        else:
            raise FileNotFoundError(f"3D file not found at: {geometry_path}")

    mesh = o3d.io.read_triangle_mesh(str(geometry_path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        print(f"Loaded triangle mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles from {geometry_path}")
        return mesh

    pcd = o3d.io.read_point_cloud(str(geometry_path))
    if len(pcd.points) > 0:
        print(f"Loaded point cloud: {len(pcd.points):,} points from {geometry_path}")
        return pcd

    raise ValueError(f"Could not load valid 3D geometry from {geometry_path}")


def view_interactive(geom, title="Thai Parasol - 3D Reconstruction"):
    print("Opening interactive 3D viewer (Press [H] for Open3D controls, [ESC] or [Q] to close)...")
    o3d.visualization.draw_geometries(
        [geom],
        window_name=title,
        width=1280,
        height=900,
        left=50,
        top=50,
    )


def render_turntable(geom, output_dir="outputs/renders", num_frames=12, image_size=(1920, 1080)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image_size[0], height=image_size[1], visible=False)
    vis.add_geometry(geom)

    # Setup studio background and shading
    opt = vis.get_render_option()
    opt.background_color = [0.08, 0.08, 0.10]  # Dark charcoal studio background
    opt.point_size = 3.0
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    ctr = vis.get_view_control()
    angle_step = 360.0 / num_frames

    print(f"Rendering {num_frames} Full HD turntable frames to {output_dir}...")
    for i in range(num_frames):
        vis.poll_events()
        vis.update_renderer()
        frame_path = output_dir / f"parasol_frame_{i:02d}.png"
        vis.capture_screen_image(str(frame_path), do_render=True)
        ctr.rotate(angle_step * 5, 0)

    vis.destroy_window()
    print(f"Successfully saved {num_frames} Full HD turntable frames -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render or view reconstructed 3D mesh / point cloud.")
    parser.add_argument("--mesh", default="outputs/multi_ring_mesh.ply", help="Path to 3D mesh or point cloud file (.ply)")
    parser.add_argument("--mode", choices=["view", "turntable"], default="view", help="view (interactive) or turntable (render frames)")
    parser.add_argument("--frames", type=int, default=12, help="Number of turntable frames to render")
    parser.add_argument("--output", default="outputs/renders", help="Output directory for rendered turntable frames")
    args = parser.parse_args()

    geometry = load_geometry(args.mesh)

    if args.mode == "view":
        view_interactive(geometry)
    else:
        render_turntable(geometry, output_dir=args.output, num_frames=args.frames)
