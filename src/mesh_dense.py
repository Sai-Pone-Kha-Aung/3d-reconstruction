"""
Stage 3B: Dense MVS Point Cloud Meshing & Surface Reconstruction
Specifically designed to turn dense multi-view stereo point clouds
(e.g., outputs/Dense Fused.ply from PatchMatch Stereo) into solid, watertight 3D meshes.

Key Features:
- Preserves CUDA PatchMatch MVS surface normals for sub-millimeter geometric accuracy
- Statistical outlier removal to strip stray floating points
- Screened Poisson Surface Reconstruction (Depth 10-12)
- Adaptive density trimming to remove surrounding phantom hull bubbles
- Taubin non-shrinking surface smoothing
- K-Nearest Neighbor (KNN) RGB color projection
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def find_default_dense_pointcloud():
    """Finds dense point cloud in project outputs, checking common names."""
    candidates = [
        Path("outputs/Dense Fused.ply"),
        Path("outputs/dense_fused.ply"),
        Path("outputs/Dense_Fused.ply"),
        Path("outputs/fused.ply"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_dense_point_cloud(ply_path):
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"Dense point cloud file not found at: {ply_path}")

    print(f"Loading dense point cloud from: {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    num_pts = len(pcd.points)
    if num_pts == 0:
        raise ValueError(f"No points found in {ply_path}. File may be corrupted or empty.")

    print(f" -> Successfully loaded {num_pts:,} dense 3D points.")
    print(f" -> Color data present: {pcd.has_colors()}")
    print(f" -> Normal data present: {pcd.has_normals()}")
    return pcd


def filter_dense_point_cloud(
    pcd,
    nb_neighbors=30,
    std_ratio=1.5,
    voxel_size=0.0,
    filter_black_bg=True,
    black_threshold=0.03,
):
    """
    Cleans statistical noise, optionally strips black background border points
    from transparent PNG reconstruction, and optionally voxel downsamples huge clouds.
    """
    # 1. Strip black background border points from transparent PNG reconstruction
    if filter_black_bg and pcd.has_colors():
        colors_arr = np.asarray(pcd.colors)
        # Identify points where all RGB channels are below threshold (near black background bleed)
        is_black = np.all(colors_arr <= black_threshold, axis=-1)
        num_black = np.sum(is_black)
        if num_black > 0:
            print(f"Filtering {num_black:,} black background border points (threshold={black_threshold})...")
            pcd = pcd.select_by_index(np.where(~is_black)[0])
            print(f" -> Points after black background filter: {len(pcd.points):,}")

    if voxel_size > 0.0:
        print(f"Voxel downsampling (voxel_size={voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f" -> Points after voxel downsampling: {len(pcd.points):,}")

    print(f"Removing statistical outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    clean_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    kept = len(clean_pcd.points)
    total = len(pcd.points)
    print(f" -> Retained {kept:,} / {total:,} inlier points ({kept / total * 100:.1f}%).")
    return clean_pcd


def ensure_normals(pcd, recompute=False, normal_radius=0.03, normal_max_nn=40):
    """Uses MVS normals if available, or estimates normals with KDTree."""
    if pcd.has_normals() and not recompute:
        print("Using existing PatchMatch MVS surface normals.")
        # Normalize to unit length
        normals = np.asarray(pcd.normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pcd.normals = o3d.utility.Vector3dVector(normals / norms)
        return pcd

    print(f"Estimating surface normals (radius={normal_radius}, max_nn={normal_max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=25)
    return pcd


def transfer_colors_knn(mesh, pcd, k=5):
    """Transfers smooth vertex colors from the dense point cloud onto the mesh."""
    if not pcd.has_colors():
        return mesh

    print(f"Projecting vertex colors from dense point cloud (KNN k={k})...")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    mesh_verts = np.asarray(mesh.vertices)
    mesh_colors = np.zeros((len(mesh_verts), 3))

    for i, v in enumerate(mesh_verts):
        _, idx, dists = kdtree.search_knn_vector_3d(v, k)
        dists = np.array(dists)
        weights = 1.0 / (dists + 1e-7)
        weights /= np.sum(weights)
        mesh_colors[i] = np.sum(pcd_colors[idx] * weights[:, None], axis=0)

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh_colors, 0.0, 1.0))
    return mesh


def build_dense_mesh(
    pcd,
    poisson_depth=11,
    trim_quantile=0.03,
    smoother="taubin",
    smooth_iterations=3,
    knn_colors=5,
):
    """Performs Poisson surface reconstruction, density trimming, and smoothing."""
    print(f"\nRunning Poisson Surface Reconstruction (Octree Depth = {poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    print(f" -> Raw mesh generated: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")

    # Trim low density phantom outer bubbles
    if trim_quantile > 0:
        densities_arr = np.asarray(densities)
        threshold = np.quantile(densities_arr, trim_quantile)
        mesh.remove_vertices_by_mask(densities_arr < threshold)
        print(f" -> Trimmed boundary vertices with density < {threshold:.3f} (quantile={trim_quantile}).")

    # Geometry cleanup
    print("Cleaning up non-manifold geometry and duplicate triangles...")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # Surface smoothing
    if smooth_iterations > 0:
        if smoother == "taubin":
            print(f"Applying Taubin non-shrinking smoothing ({smooth_iterations} iterations)...")
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        else:
            print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)

    # Color projection
    if knn_colors > 0 and pcd.has_colors():
        mesh = transfer_colors_knn(mesh, pcd, k=knn_colors)

    # Recompute vertex normals for realistic shading
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    default_input = find_default_dense_pointcloud()

    parser = argparse.ArgumentParser(
        description="Convert Dense MVS Point Cloud (e.g., 'Dense Fused.ply') into a solid 3D mesh."
    )
    parser.add_argument(
        "--input", "-i",
        default=str(default_input),
        help=f"Path to dense point cloud .ply file (default: '{default_input}')"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/dense_mesh.ply",
        help="Path for output 3D solid mesh (.ply) (default: outputs/dense_mesh.ply)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=11,
        help="Poisson octree depth: 10 (fast/smooth), 11 (sharp details, default), 12 (ultra fine)"
    )
    parser.add_argument(
        "--trim-quantile", "-t",
        type=float,
        default=0.03,
        help="Quantile threshold (0.01 - 0.05) to trim boundary air bubbles (default: 0.03)"
    )
    parser.add_argument(
        "--smoother",
        choices=["taubin", "laplacian"],
        default="taubin",
        help="Smoothing algorithm (default: taubin prevents volume shrinkage)"
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=3,
        help="Number of smoothing passes (default: 3)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsampling radius before meshing (e.g. 0.005). 0 = off (default: 0.0)"
    )
    parser.add_argument(
        "--recompute-normals",
        action="store_true",
        help="Force recomputing normals instead of using MVS normals in the PLY"
    )
    parser.add_argument(
        "--no-filter-black-bg",
        action="store_true",
        help="Disable automatic black background point filtering for transparent PNG dense clouds"
    )
    args = parser.parse_args()

    # 1. Load Dense Point Cloud
    pcd = load_dense_point_cloud(args.input)

    # 2. Outlier filtering
    clean_pcd = filter_dense_point_cloud(
        pcd,
        nb_neighbors=30,
        std_ratio=1.5,
        voxel_size=args.voxel_size,
        filter_black_bg=not args.no_filter_black_bg,
    )

    # 3. Ensure Normals
    clean_pcd = ensure_normals(clean_pcd, recompute=args.recompute_normals)

    # 4. Build Mesh
    mesh = build_dense_mesh(
        clean_pcd,
        poisson_depth=args.depth,
        trim_quantile=args.trim_quantile,
        smoother=args.smoother,
        smooth_iterations=args.smooth_iterations,
        knn_colors=5,
    )

    # 5. Save Solid Mesh
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_file), mesh)

    print("\n" + "=" * 50)
    print(f"🎉 SOLID 3D MESH SUCCESSFULLY CREATED!")
    print(f"Saved to:       {out_file}")
    print(f"Mesh Vertices:  {len(mesh.vertices):,}")
    print(f"Mesh Triangles: {len(mesh.triangles):,}")
    print("=" * 50)
    print("\nNext: To view the solid mesh, run:")
    print(f"python src/render.py --mesh \"{out_file}\" --mode view\n")
