import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ..base import TrackerBase
from ..smoothing import VectorKalmanFilter
import globals as g
from globals import OSC
import numpy as np
import cv2, json

def get_pose(landmarks, flip=True):
    pose = np.asarray([(i.x, i.y, i.z) or i for i in landmarks])
    if flip:
        pose[:, 0] = -pose[:, 0] # flip the points a bit since steamvrs coordinate system is a bit diffrent
        pose[:, 1] = -pose[:, 1]
    return pose - [
        -(landmarks[27].x + landmarks[28].x) / 2,
        -(landmarks[27].y + landmarks[28].y) / 2,
        min(landmarks[29].z, landmarks[30].z)
    ] - json.load(open('calibration.json'))

class Detector:
    def __init__(self):
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="./models/pose_landmarker_full.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.__result_callback))
        self.start_time = cv2.getTickCount()
        self.result = None
    def detect(self, mp_image):
        timestamp_ms = int((cv2.getTickCount() - self.start_time) / cv2.getTickFrequency() * 1000)
        self.landmarker.detect_async(mp_image, timestamp_ms)
        return self.result
    def __result_callback(self, detection_result, output_image, timestamp_ms):
        self.result = detection_result if detection_result.pose_landmarks else None

class TrackerBackend:
    def update(self):
        if g.settings.full_body:
            x, y, z = g.smoothed.pose[0].array * 100 / 2
            g.raw.head.position.update(x, z, y)
            for i, position in enumerate(g.smoothed.pose):
                if i <= 10 or i in [13,14,15,16,17,19,21,18,20,22]: continue
                OSC.send_message('/VMT/Room/Driver', [i+3, 1, 0.0, *position.tuple, *(.0,.0,.0,1.)])
        else:
            for i in range(33):
                if i <= 10 or i in [13,14,15,16,17,19,21,18,20,22]: continue
                OSC.send_message('/VMT/Room/Driver', [i+3, 0, 0.0, .0, .0, .0, .0, .0, .0, 1.])

class PoseTracker(TrackerBase):
    def __init__(self):
        super().__init__()
        self.detector = Detector()
        self.backend = TrackerBackend()
        self.filters = [VectorKalmanFilter(0.02, 0.05, 3) for _ in range(33)]
    def process_frame(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if result is None: return
        world_landmarks, = result.pose_world_landmarks
        g.settings.full_body = True or sum([world_landmarks[i].visibility for i in range(25, 28+1)]) > 3
        for i, e in enumerate(get_pose(world_landmarks)): g.raw.pose[i].update(*e)
        landmarks, = result.pose_landmarks
        return landmarks
    @staticmethod
    def draw_landmarks(image_rgb, landmarks):
        if landmarks is None or not g.settings.full_body: return image_rgb
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks])
        mp.solutions.drawing_utils.draw_landmarks(image_rgb, face_landmarks_proto, mp.solutions.pose.POSE_CONNECTIONS)
    def smooth_frame(self, dt):
        for i in range(33):
            self.filters[i].predict(dt)
            filtered = self.filters[i].update(g.raw.pose[i].tuple)
            diff = (filtered - g.smoothed.pose[i].array) * dt * 60
            g.smoothed.pose[i].update(*(g.smoothed.pose[i].array + diff))
