import mediapipe as mp
import joblib
from .hand import hand_pred_handling, draw_hand_landmarks
from .backend import GloveControllerSender
from ..base import TrackerBase

class HandTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.4)
        self.feature_model = joblib.load('./models/hand_feature_model.pkl')
        self.regression_model = joblib.load('./models/hand_regression_model.pkl')
        self.backend = GloveControllerSender(osc_ip='127.0.0.1', osc_port=39570)
    def process_frame(self, image_rgb):
        hand_result = self.detector.process(image_rgb)
        hand_pred_handling(self.backend, hand_result, self.feature_model, self.regression_model)
        return hand_result
    def draw_landmarks(self, image_rgb, hand_result):
        image_marked = draw_hand_landmarks(image_rgb, hand_result)
        return image_marked
