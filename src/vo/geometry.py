import numpy as np
import cv2


def make_homogeneous(R, t):
    """
    Build a 4x4 homogeneous transform from rotation R (3x3) and translation t (3x1 or 3,).
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def compose(T_global, T_rel):
    """
    Compose two SE(3) transforms: T = T_global * T_rel
    """
    return T_global @ T_rel


def triangulate_points(K, R, t, pts1, pts2):
    """
    Triangulate 3D points from two calibrated views.

    Args:
        K    : 3x3 camera intrinsics.
        R, t : relative pose of camera 2 w.r.t. camera 1 (R, t).
        pts1 : Nx2 array of 2D points in image 1.
        pts2 : Nx2 array of 2D points in image 2.

    Returns:
        X_cam1: Nx3 array of 3D points in camera-1 coordinate frame.
    """
    # Projection matrices for the two views
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # [I | 0]
    P2 = K @ np.hstack([R, t])                          # [R | t]

    # OpenCV expects 2xN arrays (float32 or float64)
    pts1_h = pts1.T  # 2 x N
    pts2_h = pts2.T  # 2 x N

    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4 x N
    X_h /= X_h[3, :]   # homogeneous normalization
    X = X_h[:3, :].T   # N x 3

    return X
