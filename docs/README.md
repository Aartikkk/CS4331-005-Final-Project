# CS4331-005 Autonomous Driving  
## Final Project: KITTI Visual Localization & Trajectory Estimation
## System Name: Monocular VO

This project implements **MonocularVO**, a simple but modular **monocular visual odometry (VO)** pipeline on a KITTI City sequence.  
The system estimates per-frame camera poses, builds a global trajectory, applies a small optimization step, and visualizes the motion.

---

## Dataset
- **KITTI Raw City dataset**
- Sequence: `2011_09_26_drive_0005_sync`
- Camera: left camera (`image_00`)
- Calibration: `calib_cam_to_cam.txt` (uses the rectified left camera matrix to get `K`)
---

## System Overview (MonocularVO)

The pipeline is modular and split into small Python files under `src/vo/`:

1. **ORB Feature Extraction & Matching**  
   - ORB keypoints (default: 2000 per frame)  
   - Brute-force matcher with Hamming distance  
   - Lowe’s ratio test + basic filtering

2. **Relative Pose Estimation (Essential Matrix)**  
   - Uses `cv2.findEssentialMat` with RANSAC and intrinsics `K`  
   - Recovers relative pose `(R, t)` with `cv2.recoverPose`  
   - This gives a robust but noisy frame-to-frame motion

3. **PnP-Based Pose Refinement**  
   - Inlier matches are triangulated using `geometry.triangulate_points`  
   - A refined pose is computed with `cv2.solvePnP` (iterative)  
   - This reduces reprojection error and stabilizes the relative pose

4. **Trajectory Composition (SE(3))**  
   - Relative poses are converted to 4×4 matrices and chained:
     `T_k = T_{k-1} * ΔT_k`  
   - Implemented in `geometry.py`

5. **3D Kalman Smoothing (Original Component)**  
   - A small 3D Kalman filter (`KalmanSmoother3D` in `smoothing.py`)  
   - Runs on translations only (rotation stays the same)  
   - Smooths frame-to-frame jitter while keeping the overall path

6. **Visualization & Statistics**  
   - `viz.py` plots:
     - Top-down X–Z trajectory with raw + smoothed path + heading arrows  
     - Clean trajectory with start/end markers and orientation arrows  
   - `optimizer.py` prints:
     - Path length (up to scale)  
     - Mean / min / max step sizes  
     - Raw vs smoothed trajectory comparison  
---

## Main Script & Config
- **Main entry point:** `scripts/run_kitti.py`  
- **Config file:** `configs/kitti_city_0005.yaml`  
  - Sets dataset path, ORB feature count, RANSAC threshold, smoothing parameters, etc.
---

## How to Run
From the project root:

```bash
python scripts/run_kitti.py --config configs/kitti_city_0005.yaml
