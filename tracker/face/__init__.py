import cv2
import mediapipe as mp
from .face import initialize_face, draw_face_landmarks
from .tongue import initialize_tongue_model
from ..base import TrackerBase

class FaceTracker(TrackerBase):
    def __init__(self):
        self.tongue_model = initialize_tongue_model()
        self.detector = initialize_face(self.tongue_model)
        self.start_time = cv2.getTickCount()
    def process_frame(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int((cv2.getTickCount() - self.start_time) * 1000 / cv2.getTickFrequency())
        self.detector.detect_async(mp_image, timestamp_ms=timestamp_ms)
        image_marked = draw_face_landmarks(image_rgb)
        return image_marked
