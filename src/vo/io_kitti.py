import os
import cv2
import numpy as np
from pathlib import Path

class KittiLoader:
    def __init__(self, sequence_path, calib_file, camera_id="image_00"):
        self.sequence_path = Path(sequence_path)
        self.calib_file = Path(calib_file)
        self.camera_id = camera_id
        
        # Debug prints
        print("CWD =", os.getcwd())
        print("sequence_path =", sequence_path)

        # image folder path
        self.img_dir = self.sequence_path / camera_id / "data"
        print("full image dir =", self.img_dir)

        # Loading all PNG images
        self.images = sorted(self.img_dir.glob("*.png"))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Loading camera intrinsics
        self.K = self._load_intrinsics()

    # KITTI calibration file parser
    def _load_intrinsics(self):
        data = {}

        with open(self.calib_file, "r") as f:
            for line in f:
                if ":" not in line:
                    continue

                key, val = line.split(":", 1)

                try:
                    nums = np.array([float(v) for v in val.split()])
                    data[key.strip()] = nums
                except ValueError:
                    continue

        if "P_rect_00" in data:
            P = data["P_rect_00"].reshape(3, 4)
        elif "P0" in data:
            P = data["P0"].reshape(3, 4)
        else:
            raise RuntimeError("Could not find P_rect_00 or P0 in calibration file.")

        # returning 3x3 intrinsic matrix
        return P[:, :3]

    # Loading individual image
    def get_image(self, idx):
        return cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.images)
