import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import joblib
from pynput import keyboard
from itertools import starmap
from .hand import hand_pred_handling
from .backend import GloveControllerSender
from ..smoothing import VectorKalmanFilter, angle_diff
from ..base import TrackerBase
import globals as g

class HandTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.4)
        self.feature_model = joblib.load('./models/hand_feature_model.pkl')
        self.regression_model = joblib.load('./models/hand_regression_model.pkl')
        self.backend = GloveControllerSender()
        self.left_hand_pos_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.left_hand_rot_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.left_hand_finger_filter = VectorKalmanFilter(0.005, 0.03, 5)
        self.right_hand_pos_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.right_hand_rot_filter = VectorKalmanFilter(0.01, 0.03, 3)
        self.right_hand_finger_filter = VectorKalmanFilter(0.005, 0.03, 5)
        self.listeners = [keyboard.Listener(on_release=self.on_keyboard_release)]
    def on_keyboard_release(self, key):
        if key != keyboard.KeyCode.from_char('`'): return
        self.backend.pointer_mode = not self.backend.pointer_mode
    def process_frame(self, image_rgb):
        hand_result = self.detector.process(image_rgb)
        hand_pred_handling(hand_result, self.feature_model, self.regression_model)
        return hand_result
    @staticmethod
    def draw_landmarks(image_rgb, detection_result):
        MARGIN = 10 # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        landmarks = detection_result.multi_hand_landmarks
        handedness = detection_result.multi_handedness
        if landmarks is None or handedness is None: return image_rgb
        for hand, hand_landmarks in zip(handedness, landmarks):
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks.landmark # Access through .landmark
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                image_rgb,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = image_rgb.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
            y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN
            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                image_rgb,
                f'{'Left' if hand.classification[0].label == 'Right' else 'Right'}', # Handedness label
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA)
    def smooth_frame(self, dt):
        self.left_hand_pos_filter.predict(dt)
        filtered = self.left_hand_pos_filter.update(g.raw.left_hand.position.tuple)
        diff = (filtered - g.smoothed.left_hand.position.array) * dt * 120
        g.smoothed.left_hand.position.update(*(g.smoothed.left_hand.position.array + diff))

        self.left_hand_rot_filter.predict(dt)
        filtered = self.left_hand_rot_filter.update(g.raw.left_hand.rotation.tuple, is_rotation=True)
        diff = np.array(list(starmap(angle_diff, zip(filtered, g.smoothed.left_hand.rotation.array)))) * dt * 90
        g.smoothed.left_hand.rotation.update(*(g.smoothed.left_hand.rotation.array + diff))

        self.left_hand_finger_filter.predict(dt)
        filtered = self.left_hand_finger_filter.update(g.raw.left_hand.blendshapes.tuple)
        diff = (filtered - g.smoothed.left_hand.blendshapes.array) * dt * 60
        g.smoothed.left_hand.blendshapes.update(*(g.smoothed.left_hand.blendshapes.array + diff))

        self.right_hand_pos_filter.predict(dt)
        filtered = self.right_hand_pos_filter.update(g.raw.right_hand.position.tuple)
        diff = (filtered - g.smoothed.right_hand.position.array) * dt * 120
        g.smoothed.right_hand.position.update(*(g.smoothed.right_hand.position.array + diff))

        self.right_hand_rot_filter.predict(dt)
        filtered = self.right_hand_rot_filter.update(g.raw.right_hand.rotation.tuple, is_rotation=True)
        diff = np.array(list(starmap(angle_diff, zip(filtered, g.smoothed.right_hand.rotation.array)))) * dt * 90
        g.smoothed.right_hand.rotation.update(*(g.smoothed.right_hand.rotation.array + diff))

        self.right_hand_finger_filter.predict(dt)
        filtered = self.right_hand_finger_filter.update(g.raw.right_hand.blendshapes.tuple)
        diff = (filtered - g.smoothed.right_hand.blendshapes.array) * dt * 60
        g.smoothed.right_hand.blendshapes.update(*(g.smoothed.right_hand.blendshapes.array + diff))
