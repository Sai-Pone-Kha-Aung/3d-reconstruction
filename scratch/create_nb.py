import json

notebook_path = 'notebooks/3dgs_mesh_extraction.ipynb'

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# 3DGS to Mesh Extraction & Evaluation\n',
            '\n',
            'This notebook demonstrates how to convert a trained **3D Gaussian Splatting (3DGS)** model (`3DGS_Result.ply`) into a clean **3D Polygonal Triangle Mesh** (`.ply` / `.obj`).\n',
            '\n',
            '### Workflow Pipeline\n',
            '1. **Parse & Filter 3DGS Attributes**: Extract 3D Gaussian centers $(x, y, z)$, evaluate opacities $\\sigma$, spherical harmonics $f_{dc}$ to RGB colors, and calculate splat scales.\n',
            '2. **Point Cloud Preprocessing**: Remove floating artifacts, filter low-opacity/outlier splats, and estimate consistent surface normals.\n',
            '3. **Mesh Reconstruction Algorithms**:\n',
            '   - **Screened Poisson Surface Reconstruction**: Water-tight, smooth surface estimation with low-density trimming.\n',
            '   - **Ball-Pivoting Algorithm (BPA)**: Delaunay-based rolling ball reconstruction preserving sharp geometric features.\n',
            '4. **Post-Processing**: Non-manifold edge cleanup, mesh smoothing, and vertex color transfer.\n',
            '5. **Interactive 3D Visualization & Export**: Interactive Plotly/Open3D inspection and export to standard formats (`.ply`, `.obj`).'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 0: Imports and Environment Setup\n',
            'import os\n',
            'import numpy as np\n',
            'import open3d as o3d\n',
            'from plyfile import PlyData\n',
            'import matplotlib.pyplot as plt\n',
            'import plotly.graph_objects as go\n',
            '\n',
            'print(f"Open3D version: {o3d.__version__}")\n',
            'print("Libraries loaded successfully!")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 1: Define File Paths\n',
            'INPUT_PLY_PATH = "../outputs/export_splat_clean 3/3DGS_Result.ply"\n',
            'OUTPUT_DIR = "../outputs/mesh_from_3dgs"\n',
            'os.makedirs(OUTPUT_DIR, exist_ok=True)\n',
            '\n',
            'assert os.path.exists(INPUT_PLY_PATH), f"File not found: {INPUT_PLY_PATH}"\n',
            'print(f"Found input 3DGS file: {INPUT_PLY_PATH} ({os.path.getsize(INPUT_PLY_PATH) / 1e6:.2f} MB)")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 2: Parse 3DGS PLY and Convert Attributes\n',
            'print("Reading 3DGS PLY...")\n',
            'plydata = PlyData.read(INPUT_PLY_PATH)\n',
            'vertex = plydata["vertex"]\n',
            'total_splats = len(vertex.data)\n',
            'print(f"Total 3D Gaussians loaded: {total_splats:,}")\n',
            '\n',
            '# 1. Splat positions (x, y, z)\n',
            'xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float64)\n',
            '\n',
            '# 2. Opacity: 3DGS stores opacity in logit space; convert with sigmoid: 1 / (1 + exp(-x))\n',
            'raw_opacity = np.asarray(vertex["opacity"]).astype(np.float64)\n',
            'opacity = 1.0 / (1.0 + np.exp(-raw_opacity))\n',
            '\n',
            '# 3. Colors: Convert 0-order spherical harmonics (f_dc) to RGB [0, 1]\n',
            'SH_C0 = 0.28209479177387814\n',
            'rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1).astype(np.float64) * SH_C0 + 0.5\n',
            'rgb = np.clip(rgb, 0.0, 1.0)\n',
            '\n',
            '# 4. Scales: 3DGS scales are stored in log-space\n',
            'scale = np.exp(np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1).astype(np.float64))\n',
            'mean_scale = np.mean(scale, axis=-1)\n',
            '\n',
            'print(f"Bounding box min: {xyz.min(axis=0)}")\n',
            'print(f"Bounding box max: {xyz.max(axis=0)}")\n',
            'print(f"Opacity percentiles (25%, 50%, 75%): {np.percentile(opacity, [25, 50, 75]).round(3)}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 3: Filter Points and Prepare Clean Point Cloud\n',
            '# Gaussians with low opacity are often semi-transparent fog or background noise\n',
            'OPACITY_THRESHOLD = 0.5\n',
            '# Gaussians with abnormally large scales are often floaters / sky artifacts\n',
            'SCALE_MAX_THRESHOLD = np.percentile(mean_scale, 98)\n',
            '\n',
            'valid_mask = (opacity >= OPACITY_THRESHOLD) & (mean_scale <= SCALE_MAX_THRESHOLD)\n',
            'print(f"Retained {np.sum(valid_mask):,} / {total_splats:,} points ({np.mean(valid_mask)*100:.1f}%)")\n',
            '\n',
            'pcd = o3d.geometry.PointCloud()\n',
            'pcd.points = o3d.utility.Vector3dVector(xyz[valid_mask])\n',
            'pcd.colors = o3d.utility.Vector3dVector(rgb[valid_mask])\n',
            '\n',
            '# Statistical Outlier Removal to eliminate remaining isolated floaters\n',
            'cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)\n',
            'pcd_clean = pcd.select_by_index(ind)\n',
            'print(f"Points after outlier removal: {len(pcd_clean.points):,}")\n',
            '\n',
            '# Estimate surface normals\n',
            'print("Estimating surface normals...")\n',
            'pcd_clean.estimate_normals(\n',
            '    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30)\n',
            ')\n',
            'pcd_clean.orient_normals_consistent_tangent_plane(k=15)'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 4: Reconstruction Method A - Screened Poisson Surface Reconstruction\n',
            'print("Running Screened Poisson Surface Reconstruction...")\n',
            '# depth=8 or 9 balances detail and speed (depth=9 is finer, depth=8 is fast)\n',
            'poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(\n',
            '    pcd_clean, depth=8, linear_fit=True\n',
            ')\n',
            'densities = np.asarray(densities)\n',
            '\n',
            '# Poisson creates a bounding hull; trim low-density vertices\n',
            'density_threshold = np.percentile(densities, 5)\n',
            'vertices_to_remove = densities < density_threshold\n',
            'poisson_mesh.remove_vertices_by_mask(vertices_to_remove)\n',
            '\n',
            '# Clean up non-manifold elements\n',
            'poisson_mesh.remove_degenerate_triangles()\n',
            'poisson_mesh.remove_duplicated_triangles()\n',
            'poisson_mesh.remove_duplicated_vertices()\n',
            'poisson_mesh.remove_non_manifold_edges()\n',
            '\n',
            'print(f"Poisson Mesh - Vertices: {len(poisson_mesh.vertices):,}, Triangles: {len(poisson_mesh.triangles):,}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 5: Reconstruction Method B - Ball Pivoting Algorithm (BPA)\n',
            'print("Running Ball-Pivoting Algorithm (BPA)...")\n',
            'distances = pcd_clean.compute_nearest_neighbor_distance()\n',
            'avg_dist = np.mean(distances)\n',
            'radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]\n',
            '\n',
            'bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(\n',
            '    pcd_clean, o3d.utility.DoubleVector(radii)\n',
            ')\n',
            '\n',
            'bpa_mesh.remove_degenerate_triangles()\n',
            'bpa_mesh.remove_duplicated_triangles()\n',
            'bpa_mesh.remove_duplicated_vertices()\n',
            'bpa_mesh.remove_non_manifold_edges()\n',
            '\n',
            'print(f"BPA Mesh - Vertices: {len(bpa_mesh.vertices):,}, Triangles: {len(bpa_mesh.triangles):,}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 6: Transfer Colors from Point Cloud to Mesh\n',
            '# Build KDTree to query nearest point colors for Poisson mesh vertices\n',
            'pcd_tree = o3d.geometry.KDTreeFlann(pcd_clean)\n',
            'mesh_vertices = np.asarray(poisson_mesh.vertices)\n',
            'mesh_colors = []\n',
            '\n',
            'pcd_colors_arr = np.asarray(pcd_clean.colors)\n',
            'for v in mesh_vertices:\n',
            '    [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)\n',
            '    mesh_colors.append(pcd_colors_arr[idx[0]])\n',
            '\n',
            'poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))\n',
            'print("Vertex colors transferred successfully to Poisson mesh!")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 7: Export Reconstructed Meshes\n',
            'poisson_out = os.path.join(OUTPUT_DIR, "3dgs_poisson_mesh.ply")\n',
            'bpa_out = os.path.join(OUTPUT_DIR, "3dgs_bpa_mesh.ply")\n',
            'poisson_obj = os.path.join(OUTPUT_DIR, "3dgs_poisson_mesh.obj")\n',
            '\n',
            'o3d.io.write_triangle_mesh(poisson_out, poisson_mesh)\n',
            'o3d.io.write_triangle_mesh(bpa_out, bpa_mesh)\n',
            'o3d.io.write_triangle_mesh(poisson_obj, poisson_mesh)\n',
            '\n',
            'print(f"Saved Poisson Mesh to: {poisson_out}")\n',
            'print(f"Saved Poisson OBJ to:  {poisson_obj}")\n',
            'print(f"Saved BPA Mesh to:     {bpa_out}")'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Step 8: Interactive 3D Mesh Inspection (Plotly)\n',
            '# Downsample mesh slightly for fast web rendering in notebook\n',
            'mesh_disp = poisson_mesh.simplify_quadric_decimation(target_number_of_triangles=20000)\n',
            'verts = np.asarray(mesh_disp.vertices)\n',
            'tris = np.asarray(mesh_disp.triangles)\n',
            'v_colors = np.asarray(mesh_disp.vertex_colors)\n',
            '\n',
            '# Convert RGB colors to hex strings for Plotly\n',
            'hex_colors = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in v_colors]\n',
            '\n',
            'fig = go.Figure(data=[\n',
            '    go.Mesh3d(\n',
            '        x=verts[:, 0],\n',
            '        y=verts[:, 1],\n',
            '        z=verts[:, 2],\n',
            '        i=tris[:, 0],\n',
            '        j=tris[:, 1],\n',
            '        k=tris[:, 2],\n',
            '        vertexcolor=hex_colors,\n',
            '        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),\n',
            '        lightposition=dict(x=100, y=200, z=100)\n',
            '    )\n',
            '])\n',
            '\n',
            'fig.update_layout(\n',
            '    title="Interactive 3D Mesh from 3DGS (Poisson Reconstruction)",\n',
            '    scene=dict(\n',
            '        xaxis=dict(visible=False),\n',
            '        yaxis=dict(visible=False),\n',
            '        zaxis=dict(visible=False),\n',
            '        aspectmode="data"\n',
            '    ),\n',
            '    margin=dict(l=0, r=0, b=0, t=40),\n',
            '    height=650\n',
            ')\n',
            'fig.show()'
        ]
    }
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3 (.venv)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.12.3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print('Clean notebook written successfully.')
