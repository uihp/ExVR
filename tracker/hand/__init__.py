from .hand import initialize_hand, hand_pred_handling, initialize_hand_depth, draw_hand_landmarks
from .backend import GloveControllerSender
from ..base import TrackerBase

class HandTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = initialize_hand()
        self.feature_model, self.regression_model = initialize_hand_depth()
        self.backend = GloveControllerSender(osc_ip='127.0.0.1', osc_port=39570)
    def process_frame(self, image_rgb):
        hand_result = self.detector.process(image_rgb)
        hand_pred_handling(self.backend, hand_result, self.feature_model, self.regression_model)
        return hand_result
    def draw_landmarks(self, image_rgb, hand_result):
        image_marked = draw_hand_landmarks(image_rgb, hand_result)
        return image_marked
