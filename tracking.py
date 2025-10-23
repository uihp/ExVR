from tracker.face import FaceTracker
from tracker.hand import HandTracker
from tracker.smoothing import Smoothing

class Tracker:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.hand_tracker = HandTracker()
        self.smoothing = Smoothing()
    def process_frames(self, image_rgb):
        self.face_tracker.process_frame(image_rgb)
        marked_image_rgb = self.face_tracker.draw_landmarks(image_rgb)
        hand_result = self.hand_tracker.process_frame(image_rgb)
        marked_image_rgb = self.hand_tracker.draw_landmarks(marked_image_rgb, hand_result)
        return marked_image_rgb
    def start(self):
        self.face_tracker.start()
        self.hand_tracker.start()
        self.smoothing.start_thread()
    def stop(self):
        self.smoothing.stop_thread()
        self.face_tracker.stop()
        self.hand_tracker.stop()
