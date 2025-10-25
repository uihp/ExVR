import numpy as np

class VectorKalmanFilter:
    '''Multi‑dimensional (vector) Kalman filter used for the high‑priority actions
    that need per‑frame smoothing. A separate instance is created per *action*,
    not per index, so that highly‑correlated channels (e.g. XYZ position) can be
    treated together.
    '''

    def __init__(self, q_process: float, r_measure: float, dim: int):
        self.dim = dim
        self.Q = np.eye(dim) * q_process  # Process‑noise covariance
        self.R = np.eye(dim) * r_measure  # Measurement‑noise covariance
        self.x = None                     # State estimate (dim,)
        self.P = np.eye(dim)              # Estimate covariance

    def predict(self, dt: float = 1.0):
        'Time‑update (prediction) step.'
        if self.x is None:
            return  # filter not initialised yet
        # Simple random‑walk model ⇒ F = I, so only P changes
        self.P += self.Q * dt

    def update(self, z, is_rotation: bool = False):
        '''Measurement‑update (correction) step.

        Args:
            z            : iterable/np.ndarray of new observations (dim,)
            is_rotation  : treat the measurements as angles in degrees and wrap
                            differences across ±180°.
        Returns:
            np.ndarray   : the updated state estimate (copy).
        '''
        z = np.asarray(z, dtype=np.float64)

        # First measurement initialises the filter
        if self.x is None:
            self.x = z.copy()
            return self.x.copy()

        # Innovation (measurement residual)
        if is_rotation:
            y = np.vectorize(angle_diff)(z, self.x)
        else:
            y = z - self.x

        # Innovation covariance & Kalman gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K) @ self.P
        return self.x.copy()

def angle_diff(current: float, target: float) -> float:
    'Minimum signed difference from *target* to *current* (deg).'
    diff = (current - target) % 360.0
    return diff - 360.0 if diff > 180.0 else diff
