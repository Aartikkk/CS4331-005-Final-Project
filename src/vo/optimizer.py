import numpy as np

def compute_path_length(traj: np.ndarray) -> float:
    """Total length of a 3D path"""
    if traj.shape[0] < 2:
        return 0.0
    diffs = traj[1:] - traj[:-1]
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(segment_lengths.sum())

def compute_step_stats(traj: np.ndarray) -> dict:
    """Basic stats on step lengths"""
    if traj.shape[0] < 2:
        return {"mean_step": 0.0, "max_step": 0.0, "min_step": 0.0}
    diffs = traj[1:] - traj[:-1]
    seg = np.linalg.norm(diffs, axis=1)
    return {
        "mean_step": float(seg.mean()),
        "max_step": float(seg.max()),
        "min_step": float(seg.min())
    }

def print_trajectory_report(traj_raw: np.ndarray, traj_smooth: np.ndarray | None = None):
    print("\nTrajectory Report")
    print(f"Frames (raw): {traj_raw.shape[0]}")
    length_raw = compute_path_length(traj_raw)
    stats_raw = compute_step_stats(traj_raw)
    print(f"Raw path length: {length_raw:.2f} (up-to-scale units)")
    print(f"Raw step mean: {stats_raw['mean_step']:.3f}, "
          f"min: {stats_raw['min_step']:.3f}, max: {stats_raw['max_step']:.3f}")

    if traj_smooth is not None and traj_smooth.shape[0] > 1:
        length_s = compute_path_length(traj_smooth)
        stats_s = compute_step_stats(traj_smooth)
        print(f"\nSmoothed path length: {length_s:.2f} (up-to-scale units)")
        print(f"Smoothed step mean: {stats_s['mean_step']:.3f}, "
              f"min: {stats_s['min_step']:.3f}, max: {stats_s['max_step']:.3f}")
        print(f"Relative path-length change: "
              f"{(length_s - length_raw) / (length_raw + 1e-8) * 100:.2f}%")
