import mediapipe as mp
import numpy as np
from .face import Detector, draw_face_landmarks
from .backend import TrackerBackend
from ..base import TrackerBase
from ..smoothing import VectorKalmanFilter, angle_diff
import globals as g
from itertools import starmap

class FaceTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = Detector()
        self.backend = TrackerBackend()
        self.head_pos_filter = VectorKalmanFilter(0.005, 0.03, 3)
        self.head_rot_filter = VectorKalmanFilter(0.02, 0.03, 3)
    def process_frame(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        self.detector.detect(mp_image)
    def draw_landmarks(self, image_rgb):
        image_marked = draw_face_landmarks(image_rgb)
        return image_marked
    def smooth_frame(self, dt):
        self.head_pos_filter.predict(dt)
        filtered = self.head_pos_filter.update(g.raw.head.position.tuple)
        diff = (filtered - g.smoothed.head.position.array) * dt * 20
        g.smoothed.head.position.update(*(g.smoothed.head.position.array + diff))

        self.head_rot_filter.predict(dt)
        filtered = self.head_rot_filter.update(g.raw.head.rotation.tuple, is_rotation=True)
        diff = np.array(list(starmap(angle_diff, zip(filtered, g.smoothed.head.rotation.array)))) * dt * 50
        g.smoothed.head.rotation.update(*(g.smoothed.head.rotation.array + diff))

        diff = (g.raw.face_pos.array - g.smoothed.face_pos.array) * dt * 100
        g.smoothed.face_pos.update(*(g.smoothed.face_pos.array + diff))
