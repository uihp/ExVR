import mediapipe as mp
import numpy as np
import joblib
from itertools import starmap
from .hand import hand_pred_handling, draw_hand_landmarks
from .backend import GloveControllerSender
from ..smoothing import VectorKalmanFilter, angle_diff
from ..base import TrackerBase
import globals as g

class HandTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.4)
        self.feature_model = joblib.load('./models/hand_feature_model.pkl')
        self.regression_model = joblib.load('./models/hand_regression_model.pkl')
        self.backend = GloveControllerSender(osc_ip='127.0.0.1', osc_port=39570)
        self.left_hand_pos_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.left_hand_rot_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.left_hand_finger_filter = VectorKalmanFilter(0.005, 0.03, 5)
        self.right_hand_pos_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.right_hand_rot_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.right_hand_finger_filter = VectorKalmanFilter(0.005, 0.03, 5)
    def process_frame(self, image_rgb):
        hand_result = self.detector.process(image_rgb)
        hand_pred_handling(self.backend, hand_result, self.feature_model, self.regression_model)
        return hand_result
    def draw_landmarks(self, image_rgb, hand_result):
        image_marked = draw_hand_landmarks(image_rgb, hand_result)
        return image_marked
    def smooth_frame(self, dt):
        self.left_hand_pos_filter.predict(dt)
        filtered = self.left_hand_pos_filter.update(g.raw.left_hand.position.tuple)
        diff = (filtered - g.smoothed.left_hand.position.array) * dt * 120
        g.smoothed.left_hand.position.update(*(g.smoothed.left_hand.position.array + diff))

        self.left_hand_rot_filter.predict(dt)
        filtered = self.left_hand_rot_filter.update(g.raw.left_hand.rotation.tuple, is_rotation=True)
        diff = np.array(list(starmap(angle_diff, zip(filtered, g.smoothed.left_hand.rotation.array)))) * dt * 90
        g.smoothed.left_hand.rotation.update(*(g.smoothed.left_hand.rotation.array + diff))

        self.left_hand_finger_filter.predict(dt)
        filtered = self.left_hand_finger_filter.update(g.raw.left_hand.blendshapes.tuple)
        diff = (filtered - g.smoothed.left_hand.blendshapes.array) * dt * 60
        g.smoothed.left_hand.blendshapes.update(*(g.smoothed.left_hand.blendshapes.array + diff))

        self.right_hand_pos_filter.predict(dt)
        filtered = self.right_hand_pos_filter.update(g.raw.right_hand.position.tuple)
        diff = (filtered - g.smoothed.right_hand.position.array) * dt * 120
        g.smoothed.right_hand.position.update(*(g.smoothed.right_hand.position.array + diff))

        self.right_hand_rot_filter.predict(dt)
        filtered = self.right_hand_rot_filter.update(g.raw.right_hand.rotation.tuple, is_rotation=True)
        diff = np.array(list(starmap(angle_diff, zip(filtered, g.smoothed.right_hand.rotation.array)))) * dt * 90
        g.smoothed.right_hand.rotation.update(*(g.smoothed.right_hand.rotation.array + diff))

        self.right_hand_finger_filter.predict(dt)
        filtered = self.right_hand_finger_filter.update(g.raw.right_hand.blendshapes.tuple)
        diff = (filtered - g.smoothed.right_hand.blendshapes.array) * dt * 60
        g.smoothed.right_hand.blendshapes.update(*(g.smoothed.right_hand.blendshapes.array + diff))
