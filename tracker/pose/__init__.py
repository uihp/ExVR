import mediapipe as mp
from ..base import TrackerBase

class PoseTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, static_image_mode=False)
    def process_frame(self, image_rgb):
        result = self.detector.process(image_rgb)
        return result.pose_landmarks
    @staticmethod
    def draw_landmarks(image_rgb, landmarks):
        mp.solutions.drawing_utils.draw_landmarks(image_rgb, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
