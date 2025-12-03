import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  

# Helper: Compute camera positions + heading vectors
def _compute_headings_from_poses(poses: list[np.ndarray]):
    """
    Given a list of 4x4 poses (world_T_cam), return:
      - positions: Nx3
      - headings: Nx3 (camera forward direction in world frame)
    Camera forward axis is +Z in camera frame.
    """
    positions = []
    headings = []
    forward_cam = np.array([0.0, 0.0, 1.0])  # camera z-axis

    for T in poses:
        p = T[:3, 3]
        R = T[:3, :3]
        fwd_world = R @ forward_cam
        positions.append(p)
        headings.append(fwd_world)

    return np.array(positions), np.array(headings)


# Plot 1: Raw + Smoothed VO + Camera Headings (X–Z plane)
def plot_trajectory_with_orientations(
    poses: list[np.ndarray],
    traj_raw: np.ndarray,
    traj_smooth: np.ndarray | None = None,
    every_n: int = 10,
    save_path: str | None = None
):
    positions, headings = _compute_headings_from_poses(poses)

    plt.figure(figsize=(10, 7))

    # Raw trajectory
    plt.plot(
        traj_raw[:, 0], traj_raw[:, 2],
        linewidth=2, color="blue", label="Raw VO"
    )

    # Smoothed trajectory
    if traj_smooth is not None:
        plt.plot(
            traj_smooth[:, 0], traj_smooth[:, 2],
            "--", linewidth=2, color="orange", label="Smoothed (Kalman)"
        )

    # Orientation arrows
    idxs = np.arange(0, positions.shape[0], every_n)
    pos_sub = positions[idxs]
    head_sub = headings[idxs]

    u = head_sub[:, 0]
    v = head_sub[:, 2]

    plt.quiver(
        pos_sub[:, 0], pos_sub[:, 2],
        u, v,
        angles="xy",
        scale_units="xy",
        scale=5,
        width=0.004,
        color="gray",
        alpha=0.6,
        label="Camera heading"
    )

    plt.title("Estimated Camera Trajectory (X-Z plane)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


# Plot 2: Clean Trajectory + Start/End + Orientation Arrows
def plot_orientations_only(
    poses: list[np.ndarray],
    every_n: int = 10,
    save_path: str | None = None
):
    positions, headings = _compute_headings_from_poses(poses)
    X = positions[:, 0]
    Z = positions[:, 2]

    plt.figure(figsize=(10, 7))

    plt.plot(X, Z, linewidth=2, color="blue", label="Trajectory")

    plt.scatter(X[0], Z[0], s=120, color="green", marker="o", label="Start")
    plt.scatter(X[-1], Z[-1], s=130, color="red", marker="X", label="End")

    idxs = np.arange(0, len(poses), every_n)
    pos_sub = positions[idxs]
    head_sub = headings[idxs]

    u = head_sub[:, 0]
    v = head_sub[:, 2]

    plt.quiver(
        pos_sub[:, 0], pos_sub[:, 2],
        u, v,
        angles="xy",
        scale_units="xy",
        scale=5,
        width=0.004,
        color="black",
        alpha=0.8,
        label="Camera orientation"
    )

    plt.title("Camera Trajectory with Orientation (X–Z plane)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


# Plot 3: Camera Frustum Visualization (2D X–Z plane)
def plot_frustums(
    poses: list[np.ndarray],
    every_n: int = 10,
    scale: float = 1.0,
    save_path: str | None = None
):
    plt.figure(figsize=(10, 7))

    positions = np.array([T[:3, 3] for T in poses])
    X = positions[:, 0]
    Z = positions[:, 2]
    plt.plot(X, Z, color="blue", linewidth=2, label="Trajectory")

    w = 0.3 * scale
    h = 0.20 * scale
    d = 0.8 * scale

    corners_cam = np.array([
        [-w, -h, d, 1],
        [ w, -h, d, 1],
        [ w,  h, d, 1],
        [-w,  h, d, 1]
    ])

    idxs = np.arange(0, len(poses), every_n)

    for i in idxs:
        T = poses[i]
        cam_pos = T[:3, 3]

        corners_world = (T @ corners_cam.T).T

        cx = corners_world[:, 0]
        cz = corners_world[:, 2]
        cam_x = cam_pos[0]
        cam_z = cam_pos[2]

        for j in range(4):
            x1, z1 = cx[j], cz[j]
            x2, z2 = cx[(j + 1) % 4], cz[(j + 1) % 4]
            plt.plot([x1, x2], [z1, z2], color="black", linewidth=1.2)
            plt.plot([cam_x, x1], [cam_z, z1], color="black", linewidth=0.9)

    plt.title("Trajectory with Camera Frustum Visualization (X–Z plane)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


# Plot 4: 3D Trajectory Plot
def plot_trajectory_3d(
    traj_raw: np.ndarray,
    traj_smooth: np.ndarray | None = None,
    save_path: str | None = None,
):
    """
    Full 3D trajectory plot using X, Y, Z coordinates.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Raw 3D path
    ax.plot(
        traj_raw[:, 0],
        traj_raw[:, 1],
        traj_raw[:, 2],
        label="Raw VO",
        linewidth=2,
        color="blue"
    )

    # Smoothed 3D path
    if traj_smooth is not None:
        ax.plot(
            traj_smooth[:, 0],
            traj_smooth[:, 1],
            traj_smooth[:, 2],
            "--",
            label="Kalman Estimate",
            linewidth=2,
            color="orange"
        )

    ax.set_title("3D Camera Trajectory (X–Y–Z)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()
