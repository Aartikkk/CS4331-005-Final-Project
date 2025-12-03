import cv2

class ORBFeature:
    def __init__(self, orb_nfeatures=2000, orb_scaleFactor=1.2, orb_levels=8):
        self.orb = cv2.ORB_create(
            nfeatures=orb_nfeatures,
            scaleFactor=orb_scaleFactor,
            nlevels=orb_levels
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, img):
        return self.orb.detectAndCompute(img, None)

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        knn = self.bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        return good
