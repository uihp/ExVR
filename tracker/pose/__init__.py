from .pose import initialize_pose, draw_pose_landmarks
from ..base import TrackerBase

class PoseTracker(TrackerBase):
    def __init__(self):
        self.detector = initialize_pose()
    def process_frame(self, image_rgb):
        self.detector.detect_async(image_rgb)
        image_marked = draw_pose_landmarks(image_rgb)
        return image_marked
