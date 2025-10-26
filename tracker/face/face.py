import mediapipe as mp
import numpy as np
import math
import cv2

import globals as g

class Detector:
    def __init__(self):
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="./models/face_landmarker.task"),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.__result_callback))
        self.start_time = cv2.getTickCount()
        self.landmarks = None
    def detect(self, mp_image):
        timestamp_ms = int((cv2.getTickCount() - self.start_time) * 1000 / cv2.getTickFrequency())
        self.landmarker.detect_async(mp_image, timestamp_ms)
        return self.landmarks
    def __result_callback(self, detection_result, output_image, timestamp_ms):
        if not detection_result.face_landmarks:
            self.landmarks = None
            return
        self.landmarks, = detection_result.face_landmarks
        trans_matrix, = detection_result.facial_transformation_matrixes
        self.__handle_result(trans_matrix)
    def __handle_result(self, trans_matrix):
        mat = np.array(trans_matrix)
        g.raw.head.position.update(
            x = -mat[0][3] * g.settings.x_scalar,
            y = -mat[2][3] * g.settings.z_scalar,
            z = mat[1][3] * g.settings.y_scalar + 50)
        g.raw.head.rotation.update(
            yaw = (
                -np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
                * 180 / math.pi
                * g.settings.yaw_scalar),
            pitch = (
                -np.arctan2(mat[2, 1], mat[2, 2])
                * 180 / math.pi
                * g.settings.pitch_scalar),
            roll = (
                -np.arctan2(mat[1, 0], mat[0, 0])
                * 180 / math.pi
                * g.settings.roll_scalar))
        g.raw.face_pos.update(self.landmarks[4].x, self.landmarks[4].y, self.landmarks[4].z)
