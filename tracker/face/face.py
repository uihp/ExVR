from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import numpy as np
import math
import cv2
from copy import deepcopy

import globals as g

def draw_face_landmarks(rgb_image):
    if g.face_landmarks is None: return rgb_image
    rgb_image = deepcopy(rgb_image)
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in g.face_landmarks])

    solutions.drawing_utils.draw_landmarks(
        image=rgb_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=rgb_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=rgb_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_iris_connections_style())

    return rgb_image

class Detector:
    def __init__(self):
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="./models/face_landmarker.task"),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=g.config["Model"]["Face"]["min_face_detection_confidence"],
            min_face_presence_confidence=g.config["Model"]["Face"]["min_face_presence_confidence"],
            min_tracking_confidence=g.config["Model"]["Face"]["min_tracking_confidence"],
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.__result_callback))
        self.start_time = cv2.getTickCount()
    def detect(self, mp_image):
        timestamp_ms = int((cv2.getTickCount() - self.start_time) * 1000 / cv2.getTickFrequency())
        self.landmarker.detect_async(mp_image, timestamp_ms)
    def __result_callback(self, detection_result, output_image, timestamp_ms):
        if not detection_result.face_landmarks:
            g.face_landmarks = None
            return
        g.face_landmarks, = detection_result.face_landmarks
        trans_matrix, = detection_result.facial_transformation_matrixes
        self.__handle_result(trans_matrix)
    def __handle_result(self, trans_matrix):
        mat = np.array(trans_matrix)

        g.raw.head.position.update(
            x = -mat[0][3] * g.settings.x_scalar,
            y = -mat[2][3] * g.settings.z_scalar,
            z = mat[1][3] * g.settings.y_scalar)
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

        head_image_position_x = g.face_landmarks[4].x
        head_image_position_y = g.face_landmarks[4].y
        head_image_position_z = g.face_landmarks[4].z

        if g.smoothing_enabled:
            if g.config["Tracking"]["Head"]["enable"]:
                # Head Position
                g.latest_data[64] = g.raw.head.position.x
                g.latest_data[65] = g.raw.head.position.y
                g.latest_data[66] = g.raw.head.position.z
                # Head Rotation
                g.latest_data[67] = g.raw.head.rotation.yaw
                g.latest_data[68] = g.raw.head.rotation.pitch
                g.latest_data[69] = g.raw.head.rotation.roll
            g.latest_data[114] = head_image_position_x
            g.latest_data[115] = head_image_position_y
            g.latest_data[116] = head_image_position_z
        else:
            if g.config["Tracking"]["Head"]["enable"]:
                # Head Position
                g.data["Position"][0]["v"] = g.raw.head.position.x
                g.data["Position"][1]["v"] = g.raw.head.position.y
                g.data["Position"][2]["v"] = g.raw.head.position.z
                # Head Rotation
                g.data["Rotation"][0]["v"] = g.raw.head.rotation.yaw
                g.data["Rotation"][1]["v"] = g.raw.head.rotation.pitch
                g.data["Rotation"][2]["v"] = g.raw.head.rotation.roll
            g.data["HeadImagePosition"][0]["v"] = head_image_position_x
            g.data["HeadImagePosition"][1]["v"] = head_image_position_y
            g.data["HeadImagePosition"][2]["v"] = head_image_position_z
