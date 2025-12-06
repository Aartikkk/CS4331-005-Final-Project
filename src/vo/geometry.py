import numpy as np
import cv2


def make_homogeneous(R, t):
    """Build a 4x4 pose matrix from R and t"""   
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


def compose(T_global, T_rel):
    """Compose two SE(3) transforms: T = T_global * T_rel"""
    return T_global @ T_rel


def triangulate_points(K, R, t, pts1, pts2):
    """Triangulate 3D points from two calibrated views"""

    # Projection matrices for the two views
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # [I | 0]
    P2 = K @ np.hstack([R, t])                          # [R | t]

    pts1_h = pts1.T  # 2 x N
    pts2_h = pts2.T  # 2 x N

    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4 x N
    X_h /= X_h[3, :]   # homogeneous normalization
    X = X_h[:3, :].T   # N x 3

    return X
