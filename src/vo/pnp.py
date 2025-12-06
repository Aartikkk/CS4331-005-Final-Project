import numpy as np
import cv2
from vo.geometry import triangulate_points


def estimate_relative_pose(kp1, kp2, matches, K, ransac_thr=1.0, prob=0.999):
    """Estimate relative pose using E + triangulation + PnP."""

    if len(matches) < 8:
        return None, None

    # matched 2D points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Step 1: Essential Matrix + RANSAC 
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=prob,
        threshold=ransac_thr
    )
    if E is None:
        return None, None

    # Mask inliers
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # Step 2: Recover Pose from E 
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
    if R is None:
        return None, None

    # Step 3: Triangulate 3D points 
    try:
        pts3D = triangulate_points(
            K, R, t,
            pts1_in.reshape(-1, 2),
            pts2_in.reshape(-1, 2)
        )
    except Exception:
        return R, t  # fall back

    # If triangulation failed
    if pts3D is None or len(pts3D) < 8:
        return R, t

    # Step 4: PnP Reprojection Optimization 
    retval, Rvec, tvec = cv2.solvePnP(
        pts3D,
        pts2_in.reshape(-1, 2),
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    R_refined, _ = cv2.Rodrigues(Rvec)
    t_refined = tvec.reshape(3, 1)

    return R_refined, t_refined
