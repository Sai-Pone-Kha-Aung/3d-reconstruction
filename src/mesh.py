"""
Stage 3: High-Definition 3D Mesh Generation & Polishing
Builds, smooths, and colors a high-fidelity 3D mesh from the sparse point cloud using:
- Octree Poisson Surface Reconstruction (Depth 9-11)
- Taubin non-shrinking surface smoothing
- K-Nearest Neighbor (KNN) inverse-distance weighted color blending
- Non-manifold geometry and degenerate triangle cleanup
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap


def load_sparse_point_cloud(sparse_dir):
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "points3D.bin").exists():
        rec_dir = sparse_dir
    elif (sparse_dir / "0" / "points3D.bin").exists():
        rec_dir = sparse_dir / "0"
    else:
        rec_dir = sparse_dir

    reconstruction = pycolmap.Reconstruction(rec_dir)
    points = reconstruction.points3D

    if not points:
        raise ValueError(f"No 3D points found in {sparse_dir}. Did reconstruct.py succeed?")

    xyz = np.array([p.xyz for p in points.values()])
    rgb = np.array([p.color / 255.0 for p in points.values()])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    print(f"Loaded point cloud with {len(pcd.points):,} raw 3D points from {rec_dir}.")
    return pcd


def filter_point_cloud(pcd, nb_neighbors=25, std_ratio=2.0):
    print(f"Filtering statistical outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Kept {len(cl.points):,} / {len(pcd.points):,} inliers after statistical filtering.")
    return cl


def transfer_colors_knn_weighted(mesh, pcd, k=5):
    """Smoothly transfers blended vertex colors from point cloud using inverse-distance weighting."""
    print(f"Transferring smooth vertex colors (KNN k={k} inverse-distance weighted blending)...")
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    mesh_verts = np.asarray(mesh.vertices)
    mesh_colors = np.zeros((len(mesh_verts), 3))

    for i, v in enumerate(mesh_verts):
        _, idx, dists = kdtree.search_knn_vector_3d(v, k)
        dists = np.array(dists)
        weights = 1.0 / (dists + 1e-6)
        weights /= np.sum(weights)
        mesh_colors[i] = np.sum(pcd_colors[idx] * weights[:, None], axis=0)

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh_colors, 0.0, 1.0))
    return mesh


def build_mesh(
    pcd,
    poisson_depth=10,
    density_trim_quantile=0.02,
    smoother="taubin",
    smooth_iterations=5,
    normal_radius=0.04,
    normal_max_nn=40,
    knn_colors=5,
):
    # 1. High precision normal estimation
    print(f"Estimating normals (radius={normal_radius}, max_nn={normal_max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=25)

    # 2. Poisson reconstruction at target octree depth
    print(f"Running Poisson surface reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    # 3. Density trimming to remove bounding artifacts without clipping object edges
    if density_trim_quantile > 0:
        densities = np.asarray(densities)
        threshold = np.quantile(densities, density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)
        print(f"Trimmed low-density boundary vertices (threshold={threshold:.3f}, quantile={density_trim_quantile})")

    # 4. Geometry cleanup
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # 5. Surface smoothing (Taubin prevents volume shrinkage)
    if smooth_iterations > 0:
        if smoother == "taubin":
            print(f"Applying Taubin non-shrinking smoothing ({smooth_iterations} iterations)...")
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        else:
            print(f"Applying Laplacian smoothing ({smooth_iterations} iterations)...")
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)

    # 6. Smooth color transfer from point cloud
    if knn_colors > 0:
        mesh = transfer_colors_knn_weighted(mesh, pcd, k=knn_colors)

    # 7. Compute smooth vertex normals
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and polish HD 3D mesh from sparse SfM point cloud.")
    parser.add_argument("--sparse", default="data/colmap_output/multi_ring_1/sparse", help="COLMAP sparse folder")
    parser.add_argument("--output", default="outputs/multi_ring_1_HD_mesh.ply", help="Output 3D mesh path (.ply)")
    parser.add_argument("--depth", type=int, default=10, help="Poisson depth (9 = fast/smooth, 10 = fine details, 11 = ultra-HD)")
    parser.add_argument("--trim-quantile", type=float, default=0.02, help="Trimming quantile (0.01-0.03 preserves delicate edges)")
    parser.add_argument("--smoother", choices=["taubin", "laplacian"], default="taubin", help="Smoothing algorithm (taubin prevents shrinkage)")
    parser.add_argument("--smooth-iterations", type=int, default=5, help="Smoothing iterations")
    parser.add_argument("--knn-colors", type=int, default=5, help="K-nearest neighbors for smooth color interpolation")
    parser.add_argument("--normal-radius", type=float, default=0.04, help="Normal estimation radius")
    args = parser.parse_args()

    # 1. Load point cloud
    pcd = load_sparse_point_cloud(args.sparse)

    # 2. Filter outliers
    clean_pcd = filter_point_cloud(pcd, nb_neighbors=25, std_ratio=2.0)

    # 3. Save cleaned point cloud
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd_out = out_path.with_name(f"{out_path.stem.replace('_mesh', '')}_pointcloud.ply")
    o3d.io.write_point_cloud(str(pcd_out), clean_pcd)
    print(f"Clean point cloud saved to: {pcd_out}")

    # 4. Generate high-definition mesh
    mesh = build_mesh(
        clean_pcd,
        poisson_depth=args.depth,
        density_trim_quantile=args.trim_quantile,
        smoother=args.smoother,
        smooth_iterations=args.smooth_iterations,
        normal_radius=args.normal_radius,
        knn_colors=args.knn_colors,
    )
    o3d.io.write_triangle_mesh(str(out_path), mesh)

    print(f"\n==========================================")
    print(f"🎉 HD Mesh saved to: {out_path}")
    print(f"Mesh Vertices:  {len(mesh.vertices):,}")
    print(f"Mesh Triangles: {len(mesh.triangles):,}")
    print(f"==========================================")