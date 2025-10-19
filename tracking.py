from tracker.face import FaceTracker
from tracker.hand import HandTracker
from smoothing import Smoothing
from copy import deepcopy

class Tracker:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.hand_tracker = HandTracker()
        self.hand_tracker.start_sending()
        self.smoothing = Smoothing()
        self.smoothing.start_thread()
    def process_frames(self, image_rgb):
        self.face_tracker.process_frame(image_rgb)
        marked_image_rgb = self.face_tracker.draw_landmarks(deepcopy(image_rgb))
        hand_result = self.hand_tracker.process_frame(image_rgb)
        marked_image_rgb = self.hand_tracker.draw_landmarks(marked_image_rgb, hand_result)
        return marked_image_rgb
    def stop(self):
        self.smoothing.stop_thread()
        self.hand_tracker.stop_sending()
