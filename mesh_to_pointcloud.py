"""
Phase 1: Mesh -> point cloud sampling for segmented component OBJ files.

Samples surface points from each segmented component mesh (Poisson-disk,
so points are roughly evenly spaced rather than density-biased toward
large flat faces). Saves per-component point clouds as .npy (Nx3, or
Nx6 if you keep normals) and optionally combines all components present
in a scene into one merged cloud for whole-scene collision checking.

Expects a nested dataset layout, one subfolder per sample/phone:
    dataset/
      1/
        model_battery.obj
        model_camera.obj
        model_Other.obj
        model_whole.obj   <- full assembled phone, treated as scene, not a component
      2/
        ...

Usage:
    python mesh_to_pointcloud.py \
        --dataset_root ./dataset \
        --output_dir ./point_clouds \
        --num_points 2048 \
        --keep_normals \
        --combine_scene

Requires: open3d, numpy
    pip install open3d numpy --break-system-packages
"""

import argparse
import os
import glob
import numpy as np
import open3d as o3d


def sample_component(obj_path, num_points, keep_normals=True):
    """Load one segmented component OBJ and Poisson-disk sample its surface."""
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"No triangles found in {obj_path} — check the mesh export.")

    mesh.compute_vertex_normals()

    # Poisson-disk sampling gives more uniform spatial coverage than
    # plain random surface sampling, which matters for a fixed-size
    # network input (you don't want all your points clustered on one
    # big flat face and none on a thin bracket).
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    points = np.asarray(pcd.points, dtype=np.float32)
    if keep_normals:
        normals = np.asarray(pcd.normals, dtype=np.float32)
        return np.concatenate([points, normals], axis=1), pcd  # Nx6
    return points, pcd  # Nx3


def sanity_check_visualize(pcd, mesh_path):
    """Quick visual sanity check — sampled cloud vs original mesh wireframe.
    Run this on a handful of components before batch-processing everything."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    pcd_vis = o3d.geometry.PointCloud(pcd)
    pcd_vis.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([mesh, pcd_vis])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help="Root dataset folder containing one subfolder per "
                              "sample (e.g. dataset/1, dataset/2, ...), each with "
                              "model_<component>.obj files")
    parser.add_argument("--output_dir", required=True,
                         help="Where to save per-sample, per-component .npy point clouds "
                              "(mirrors the dataset_root subfolder structure)")
    parser.add_argument("--num_points", type=int, default=2048,
                         help="Points per component (2048 matched GraspFC-PTX's budget)")
    parser.add_argument("--keep_normals", action="store_true",
                         help="Save surface normals alongside XYZ (Nx6 instead of Nx3)")
    parser.add_argument("--include_whole", action="store_true",
                         help="Also sample model_whole.obj (the full assembled phone). "
                              "Off by default since 'whole' is the assembly, not a "
                              "single component — turn on only if you specifically "
                              "need a full-assembly point cloud for something.")
    parser.add_argument("--combine_scene", action="store_true",
                         help="Also save one merged point cloud per sample, combining "
                              "all its components (for whole-scene collision checking, "
                              "if that's the scope you land on in Phase 0)")
    parser.add_argument("--visualize_first", action="store_true",
                         help="Pop up a viewer for the very first component processed, "
                              "as a one-time sanity check")
    args = parser.parse_args()

    sample_dirs = sorted(
        d for d in glob.glob(os.path.join(args.dataset_root, "*"))
        if os.path.isdir(d)
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No sample subfolders found in {args.dataset_root}")

    visualized_once = False
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        obj_paths = sorted(glob.glob(os.path.join(sample_dir, "model_*.obj")))

        if not args.include_whole:
            obj_paths = [p for p in obj_paths
                         if os.path.basename(p).lower() != "model_whole.obj"]

        if not obj_paths:
            print(f"[skip] sample {sample_id}: no matching model_*.obj files found")
            continue

        out_sample_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(out_sample_dir, exist_ok=True)

        all_points_for_scene = []
        for obj_path in obj_paths:
            comp_name = os.path.splitext(os.path.basename(obj_path))[0]
            try:
                sampled, pcd = sample_component(obj_path, args.num_points, args.keep_normals)
            except ValueError as e:
                print(f"[skip] sample {sample_id} / {comp_name}: {e}")
                continue

            out_path = os.path.join(out_sample_dir, f"{comp_name}.npy")
            np.save(out_path, sampled)
            print(f"sample {sample_id} / {comp_name}: saved {sampled.shape} -> {out_path}")

            if args.visualize_first and not visualized_once:
                sanity_check_visualize(pcd, obj_path)
                visualized_once = True

            if args.combine_scene:
                all_points_for_scene.append(sampled[:, :3])  # XYZ only for merged cloud

        if args.combine_scene and all_points_for_scene:
            merged = np.concatenate(all_points_for_scene, axis=0)
            merged_path = os.path.join(out_sample_dir, "_scene_combined.npy")
            np.save(merged_path, merged)
            print(f"sample {sample_id}: combined scene cloud {merged.shape} -> {merged_path}")


if __name__ == "__main__":
    main()