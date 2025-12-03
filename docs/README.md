# CS4331-005 Autonomous Driving  
## Final Project: KITTI Visual Localization & Trajectory Estimation

This project implements a complete **Visual Odometry (VO)** pipeline on a KITTI City sequence.  
The objective was to build a modular VO system that estimates per-frame camera poses, constructs a global trajectory, applies trajectory optimization, and visualizes motion using multiple plots.

## Dataset Used

- **KITTI Raw City Dataset**
- Sequence: **2011_09_26_drive_0005_sync**
- Camera: **Left color camera (image_00)**
- Calibration: Taken from `calib_cam_to_cam.txt`

This sequence does **not** include official ground-truth poses (only the Odometry dataset 00–10 provides GT), so evaluation is done visually and statistically.

---

## Features & Methods Used

### **1. ORB Feature Extraction + Matching**
- ORB keypoints (n=2000)
- Brute-force Hamming matcher
- Lowe’s ratio test

### **2. Relative Pose Estimation**
- Essential Matrix with RANSAC
- Pose recovery (R, t)
- Global pose chaining using 4×4 homogeneous matrices

### **3. Trajectory Optimization (My Original Component)**
I implemented a **3D Kalman Smoothing filter** to reduce noise in the raw VO trajectory.  
This smooths out jitter while keeping the motion realistic.

### **4. Visualization**
Two separate trajectory plots are generated:

#### **A) Raw + Smoothed + Camera Headings**  
`saved in results/figs/trajectory_full.png`

#### **B) Clean Trajectory with Start/End + Camera Orientation**  
`saved in results/figs/trajectory_orientations_only.png`

This second plot is specifically for showing orientation direction along the path.

---

## How to Run the Project

From the project root:

```bash
python scripts/run_kitti.py --config configs/kitti_city_0005.yaml
