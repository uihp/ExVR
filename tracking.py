from tracker.pose import PoseTracker
from tracker.face import FaceTracker
from tracker.hand import HandTracker
from copy import deepcopy

class Tracker:
    def __init__(self):
        self.face_tracker = FaceTracker() # [issue] face tracker needs to be loaded first
        self.pose_tracker = PoseTracker()
        self.hand_tracker = HandTracker()
    def process_frames(self, image_rgb):
        marked_image_rgb = deepcopy(image_rgb)
        landmarks = self.pose_tracker.process_frame(image_rgb)
        self.pose_tracker.draw_landmarks(marked_image_rgb, landmarks)
        landmarks = self.face_tracker.process_frame(image_rgb)
        self.face_tracker.draw_landmarks(marked_image_rgb, landmarks)
        result = self.hand_tracker.process_frame(image_rgb)
        self.hand_tracker.draw_landmarks(marked_image_rgb, result)
        return marked_image_rgb
    def start(self):
        self.face_tracker.start()
        self.hand_tracker.start()
    def stop(self):
        self.face_tracker.stop()
        self.hand_tracker.stop()
