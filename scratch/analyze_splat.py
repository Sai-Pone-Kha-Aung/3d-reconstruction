from plyfile import PlyData
import numpy as np

print("Loading PLY...")
ply = PlyData.read('outputs/export_splat_clean 3/3DGS_Result.ply')
vertex = ply['vertex']
xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
rgb = 0.28209479177387814 * f_dc + 0.5
brightness = rgb.mean(axis=-1)
opacity_logit = np.array(vertex['opacity'])
opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

print("Total vertices:", len(xyz))

# Canopy points: blue
blue_pts = xyz[(rgb[:, 2] > rgb[:, 0]) & (rgb[:, 2] > rgb[:, 1]) & (brightness > 0.3)]
center = np.median(blue_pts, axis=0)
print(f"Canopy center: {center}")

# PCA on subset of blue points to be instant
cov = np.cov(blue_pts.T)
eigvals, eigvecs = np.linalg.eigh(cov)
# eigvecs[:, 0] is the normal to the canopy plane (smallest eigenvalue)
normal = eigvecs[:, 0]
if normal[2] < 0:
    normal = -normal
u_axis = eigvecs[:, 1]
v_axis = eigvecs[:, 2]

print("Normal axis:", normal)
print("Eigenvalues (variances along normal, u, v):", eigvals)

# Project all points:
proj_n = np.dot(xyz - center, normal)
proj_u = np.dot(xyz - center, u_axis)
proj_v = np.dot(xyz - center, v_axis)
r_plane = np.sqrt(proj_u**2 + proj_v**2)

blue_mask = (rgb[:, 2] > rgb[:, 0]) & (rgb[:, 2] > rgb[:, 1]) & (brightness > 0.3)
print(f"Blue points: r_plane in [{r_plane[blue_mask].min():.4f}, {r_plane[blue_mask].max():.4f}], proj_n in [{proj_n[blue_mask].min():.4f}, {proj_n[blue_mask].max():.4f}]")

dark_mask = brightness < 0.2
print(f"Dark points: total={dark_mask.sum()}, r_plane in [{r_plane[dark_mask].min():.4f}, {r_plane[dark_mask].max():.4f}], proj_n in [{proj_n[dark_mask].min():.4f}, {proj_n[dark_mask].max():.4f}]")

# Let's see how many dark points are outside r_plane > 0.14
print(f"Dark points with r_plane > 0.14: {(dark_mask & (r_plane > 0.14)).sum()}")
print(f"Dark points with proj_n < -0.05 or proj_n > 0.05: {(dark_mask & ((proj_n < -0.05) | (proj_n > 0.05))).sum()}")

# What are the dark points with r_plane <= 0.14?
inside_dark = dark_mask & (r_plane <= 0.14)
print(f"Dark points inside r_plane <= 0.14: {inside_dark.sum()}")
print(f"Their proj_n distribution: min={proj_n[inside_dark].min():.4f}, p10={np.percentile(proj_n[inside_dark], 10):.4f}, median={np.median(proj_n[inside_dark]):.4f}, p90={np.percentile(proj_n[inside_dark], 90):.4f}, max={proj_n[inside_dark].max():.4f}")
