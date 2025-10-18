import threading
import cv2
import mediapipe as mp
from tracker.face.tongue import initialize_tongue_model
from tracker.face.face import initialize_face
from tracker.hand.hand import initialize_hand,hand_pred_handling,initialize_hand_depth
from tracker.pose.pose import initialize_pose
from utils.sender import data_send_thread
from utils.smoothing import apply_smoothing
import utils.globals as g

class Tracker:
    def __init__(self):
        if not g.model_loaded:
            print("Initializing tongue model")
            self.tongue_model = initialize_tongue_model()
            print("Initializing MediaPipe")
            self.face_detector = initialize_face(self.tongue_model)
            self.hand_detector = initialize_hand()
            self.hand_feature_model, self.hand_regression_model = initialize_hand_depth()
            self.pose_detector = initialize_pose()
            g.model_loaded = True
        self.data_thread = threading.Thread(target=data_send_thread, daemon=True)
        self.data_thread.start()
        self.smoothing_thread = threading.Thread(target=apply_smoothing, daemon=True)
        self.smoothing_thread.start()
    def process_frames(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int((cv2.getTickCount() - g.start_time) * 1000 / cv2.getTickFrequency())
        self.face_detector.detect_async(mp_image, timestamp_ms=timestamp_ms)
        self.pose_detector.detect_async(image_rgb)
        hand_result = self.hand_detector.process(image_rgb)
        hand_pred_handling(hand_result, self.hand_feature_model, self.hand_regression_model)
    def stop(self):
        g.stop_event.set()
        self.smoothing_thread.join()
        self.data_thread.join()
        g.stop_event.clear()
