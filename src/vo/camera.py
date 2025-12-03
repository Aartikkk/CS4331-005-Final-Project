"""Camera models and helpers (placeholder)."""

class Camera:
    def __init__(self, fx=0.0, fy=0.0, cx=0.0, cy=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def intrinsics(self):
        return (self.fx, self.fy, self.cx, self.cy)
