import sys
import os
import argparse
import yaml
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from vo.io_kitti import KittiLoader
from vo.feature import ORBFeature
from vo.pnp import estimate_relative_pose
from vo.geometry import make_homogeneous, compose
from vo.smoothing import KalmanSmoother3D
from vo.optimizer import print_trajectory_report
from vo.viz import (
    plot_trajectory_with_orientations,
    plot_orientations_only,
    plot_frustums,
    plot_trajectory_3d    
)


def main(cfg):

    # Loading KITTI sequence
    loader = KittiLoader(
        cfg["sequence_path"],
        cfg["calib_file"],
        cfg.get("camera_id", "image_00")
    )

    print("\nDataset Info")
    print(f"Sequence path : {cfg['sequence_path']}")
    print(f"Camera ID     : {cfg.get('camera_id', 'image_00')}")
    print(f"Total frames  : {len(loader)}")
    print("\nCamera intrinsics K:\n", loader.K)

    # Feature extraction 
    feat_cfg = cfg.get("feature", {})
    feature = ORBFeature(
        orb_nfeatures=feat_cfg.get("orb_nfeatures", 2000),
        orb_scaleFactor=feat_cfg.get("orb_scaleFactor", 1.2),
        orb_levels=feat_cfg.get("orb_levels", 8),
    )

    # Kalman smoothing
    smoother = None
    if cfg.get("use_kalman_smoothing", True):
        smoother = KalmanSmoother3D()
        print("\nUsing Kalman smoothing.")

    # RANSAC settings
    ransac_cfg = cfg.get("ransac", {})
    ransac_thr = ransac_cfg.get("threshold", 1.0)
    ransac_prob = ransac_cfg.get("prob", 0.999)

    # VO loop
    T_global = np.eye(4)
    poses = [T_global.copy()]

    traj_raw = [T_global[:3, 3].copy()]
    traj_smooth = []

    match_counts = []
    x_smooth = None

    print("\nStarting VO...")

    for i in range(len(loader) - 1):

        img1 = loader.get_image(i)
        img2 = loader.get_image(i + 1)

        kp1, desc1 = feature.extract(img1)
        kp2, desc2 = feature.extract(img2)
        matches = feature.match(desc1, desc2)

        match_counts.append(len(matches))

        if len(matches) < 8:
            print(f"Frame {i}: insufficient matches ({len(matches)})")
            continue

        R, t = estimate_relative_pose(
            kp1, kp2, matches, loader.K,
            ransac_thr=ransac_thr, prob=ransac_prob, 
        )

        if R is None:
            print(f"Frame {i}: pose estimation failed")
            continue

        # Compose global pose
        T_rel = make_homogeneous(R, t)
        T_global = compose(T_global, T_rel)
        poses.append(T_global.copy())

        p = T_global[:3, 3]
        traj_raw.append(p.copy())

        # Kalman smoothing
        if smoother:
            if x_smooth is None:
                smoother.reset()
                x_smooth = smoother.step(p)
            else:
                x_smooth = smoother.step(p)
            traj_smooth.append(x_smooth.copy())

    traj_raw = np.array(traj_raw)

    traj_smooth_arr = None
    if smoother and len(traj_smooth) > 0:
        traj_smooth_arr = np.vstack(traj_smooth)

    # VO Statistics
    print("\nVO Statistics")
    print(f"Avg matches/frame: {np.mean(match_counts):.1f}")
    print(f"Min matches      : {np.min(match_counts)}")
    print(f"Max matches      : {np.max(match_counts)}")

    print_trajectory_report(traj_raw, traj_smooth_arr)

    # Visualization 
    figs_dir = os.path.join(PROJECT_ROOT, "results", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # 1. Raw + Smoothed + Orientation
    fig1 = os.path.join(figs_dir, "trajectory_full.png")
    plot_trajectory_with_orientations(
        poses=poses,
        traj_raw=traj_raw,
        traj_smooth=traj_smooth_arr,
        every_n=10,
        save_path=fig1
    )

    # 2. Orientation-only plot
    fig2 = os.path.join(figs_dir, "trajectory_orientations_only.png")
    plot_orientations_only(
        poses=poses,
        every_n=10,
        save_path=fig2
    )

    # 3. Frustum visualization
    fig3 = os.path.join(figs_dir, "trajectory_frustums.png")
    plot_frustums(
        poses=poses,
        every_n=10,
        scale=1.0,
        save_path=fig3
    )

    # 4. 3D Trajectory Plot (X–Y–Z)
    fig3d = os.path.join(figs_dir, "trajectory_3d.png")
    plot_trajectory_3d(
        traj_raw=traj_raw,
        traj_smooth=traj_smooth_arr,
        save_path=fig3d
    )

    # Save trajectories
    results_dir = os.path.join(PROJECT_ROOT, "results")
    np.savetxt(os.path.join(results_dir, "traj_raw.txt"), traj_raw)

    if traj_smooth_arr is not None:
        np.savetxt(os.path.join(results_dir, "traj_smooth.txt"), traj_smooth_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
