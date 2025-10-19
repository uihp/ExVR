from tracker.face import FaceTracker
from tracker.hand import HandTracker
from tracker.pose import PoseTracker
from smoothing import Smoothing

class Tracker:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.pose_tracker = PoseTracker()
        self.hand_tracker = HandTracker()
        self.hand_tracker.start_sending()
        self.smoothing = Smoothing()
        self.smoothing.start_thread()
    def process_frames(self, image_rgb):
        image_rgb = self.face_tracker.process_frame(image_rgb)
        image_rgb = self.pose_tracker.process_frame(image_rgb)
        image_rgb = self.hand_tracker.process_frame(image_rgb)
        return image_rgb
    def stop(self):
        self.smoothing.stop_thread()
        self.hand_tracker.stop_sending()
