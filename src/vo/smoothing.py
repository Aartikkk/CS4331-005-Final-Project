import numpy as np

class KalmanSmoother3D:
    """
    Simple 3D Kalman smoother for translation
    State: x = [X, Y, Z]^T
    Model: x_k = A x_{k-1} + w,  z_k = C x_k + v
    Here A = I, C = I  (random walk model).
    """
    def __init__(self, process_var=0.01, meas_var=0.05):
        self.A = np.eye(3)
        self.C = np.eye(3)

        self.Q = np.eye(3) * process_var   # process noise covariance
        self.R = np.eye(3) * meas_var      # measurement noise covariance

        self.P = np.eye(3) * 0.1           # initial covariance
        self.x = np.zeros(3)               # initial state

        self.initialized = False

    def reset(self):
        self.P = np.eye(3) * 0.1
        self.x = np.zeros(3)
        self.initialized = False

    def step(self, z):
        """One predict+update step given a 3D measurement."""

        z = np.asarray(z).reshape(3)

        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.copy()

        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update
        S = self.C @ P_pred @ self.C.T + self.R       # innovation covariance
        K = P_pred @ self.C.T @ np.linalg.inv(S)      # Kalman gain

        y = z - (self.C @ x_pred)                     # innovation
        self.x = x_pred + K @ y                       # updated state
        self.P = (np.eye(3) - K @ self.C) @ P_pred    # updated covariance

        return self.x.copy()
