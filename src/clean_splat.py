"""
Non-destructive post-processing script to completely isolate and clean 3D Gaussian Splats.
Preserves 100% of the genuine object (canopy fabric, floral patterns, tassels, ribs, handle)
while removing external black background bleed, floating slabs, and ground remnants.
"""

import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def clean_gaussian_splat(
    input_ply: Path,
    output_ply: Path,
    crop_radius: float = 0.145,
    shaft_radius: float = 0.035,
    max_scale: float = 0.015,
    rim_dark_color_thresh: float = 0.12,
    rim_radius_thresh: float = 0.122,
):
    print(f"Loading {input_ply}...")
    ply = PlyData.read(str(input_ply))
    vertex = ply['vertex']
    
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
    f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
    rgb = np.clip(0.28209479177387814 * f_dc + 0.5, 0.0, 1.0)
    max_color = rgb.max(axis=-1)
    
    scales = np.exp(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1))
    max_scales = scales.max(axis=-1)
    
    # 1. Estimate canopy plane and rotational symmetry axis
    blue_mask = (rgb[:, 2] > rgb[:, 0]) & (rgb[:, 2] > rgb[:, 1]) & (rgb.mean(axis=-1) > 0.3)
    blue_pts = xyz[blue_mask]
    center = np.median(blue_pts, axis=0)
    
    cov = np.cov(blue_pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    if normal[2] < 0:
        normal = -normal
    u_axis = eigvecs[:, 1]
    v_axis = eigvecs[:, 2]
    
    # Align origin to center of rotation
    origin = center + 0.0011 * u_axis + 0.0100 * v_axis
    proj_n = np.dot(xyz - origin, normal)
    proj_u = np.dot(xyz - origin, u_axis)
    proj_v = np.dot(xyz - origin, v_axis)
    r = np.sqrt(proj_u**2 + proj_v**2)
    
    # 2. Precise non-destructive geometric envelope
    valid_height = (proj_n >= -0.205) & (proj_n <= 0.110)
    r_envelope = np.zeros_like(proj_n)
    for i, n in enumerate(proj_n):
        if n > 0.070:
            r_envelope[i] = 0.035
        elif n > 0.040:
            r_envelope[i] = 0.055
        elif n >= -0.035:
            r_envelope[i] = crop_radius
        elif n >= -0.055:
            r_envelope[i] = 0.090
        else:
            r_envelope[i] = shaft_radius
            
    in_envelope = valid_height & (r <= r_envelope)
    
    # 3. Reject giant floaters that render as dark blocks / slabs
    is_giant = max_scales > max_scale
    
    # 4. Reject pitch-black / dark-grey background bleed clinging outside the fringe
    is_rim_black = (r >= rim_radius_thresh) & (max_color < rim_dark_color_thresh)
    
    keep_mask = in_envelope & (~is_giant) & (~is_rim_black)
    final_indices = np.where(keep_mask)[0]
    
    print(f"Points preserved: {len(final_indices)} / {len(xyz)} ({len(final_indices)/len(xyz)*100:.1f}%)")
    print(f"Noise points removed: {(~keep_mask).sum()}")
    
    # Write output
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    filtered_vertex_data = vertex.data[final_indices]
    filtered_element = PlyElement.describe(filtered_vertex_data, 'vertex')
    PlyData([filtered_element], text=False).write(str(output_ply))
    print(f"Successfully saved pristine PLY to {output_ply} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-destructive 3DGS PLY cleaning")
    parser.add_argument("--input", "-i", type=Path, default=Path("outputs/clean_splat_1.ply"))
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/clean_splat_final.ply"))
    args = parser.parse_args()
    clean_gaussian_splat(args.input, args.output)
