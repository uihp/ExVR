import numpy as np
from scipy.spatial.transform import Rotation as R

import globals as g

def get_hand_pose(landmarks, reverse_flag=True):
    hand_pose = np.asarray([[l.x, l.y, l.z] for l in landmarks])
    if reverse_flag:
        hand_pose[:, 0] = -hand_pose[:, 0] # flip the points a bit since steamvrs coordinate system is a bit diffrent
        hand_pose[:, 1] = -hand_pose[:, 1]
    return hand_pose

def calc_angle(vec1, vec2):
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = np.linalg.norm(np.cross(vec1, vec2))
        dot = np.dot(vec1, vec2)
        return np.degrees(np.abs(np.arctan2(cross, dot)))

def finger_handling(hand_pose):
    finger_curl = {}
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for name in finger_names:
        cfg = {
            "thumb_base": [0, 2],
            "thumb_tip": [3, 4],
            "thumb_min": 30,
            "thumb_max": 70,
            "index_base": [0, 5],
            "index_tip": [7, 8],
            "index_min": 80,
            "index_max": 120,
            "middle_base": [0, 9],
            "middle_tip": [11, 12],
            "middle_min": 70,
            "middle_max": 120,
            "ring_base": [0, 13],
            "ring_tip": [15, 16],
            "ring_min": 70,
            "ring_max": 120,
            "pinky_base": [0, 17],
            "pinky_tip": [19, 20],
            "pinky_min": 70,
            "pinky_max": 120
        }
        base_start, base_end = cfg[f"{name}_base"]
        tip_start, tip_end = cfg[f"{name}_tip"]
        min_val = cfg[f"{name}_min"]
        max_val = cfg[f"{name}_max"]
        base_vec = hand_pose[base_end] - hand_pose[base_start]
        tip_vec = hand_pose[tip_end] - hand_pose[tip_start]
        if np.linalg.norm(base_vec) < 1e-6 or np.linalg.norm(tip_vec) < 1e-6:
            raw_angle = 0.0
        else:
            raw_angle = calc_angle(base_vec, tip_vec)
        clamped = np.clip(raw_angle, min_val, max_val)
        range_val = max_val - min_val
        norm_value = 1 - (clamped - min_val) / range_val
        norm_value = np.clip(norm_value, 0.1, 1.0)
        finger_curl[name] = round(norm_value,1)
    return finger_curl

def hand_pred_handling(detection_result, hand_feature_model, hand_regression_model):
    if detection_result.multi_handedness is None:
        g.settings.handedness = [False, False]
        return

    handedness = [False, False]
    for hand, hand_landmarks, hand_world_landmarks in zip(detection_result.multi_handedness, detection_result.multi_hand_landmarks, detection_result.multi_hand_world_landmarks):
        hand_name = 'Left' if hand.classification[0].label == 'Right' else 'Right'
        if hand_name == 'Left': handedness[0] = True
        elif hand_name == 'Right': handedness[1] = True
        world_landmarks = hand_world_landmarks.landmark
        hand_pose = get_hand_pose(world_landmarks)
        image_landmarks = hand_landmarks.landmark
        image_hand_pose = get_hand_pose(image_landmarks, False)
        hand_position = g.smoothed.face_pos.array - image_hand_pose[9] # - image_hand_pose[2]
        hand_position[:2] *= [g.settings.hand_x_scalar, g.settings.hand_y_scalar]

        keypoints = [5, 9, 13]
        data = image_hand_pose - image_hand_pose[0]
        data = data[keypoints].flatten()
        data = np.array(data)
        data = data.reshape(1, -1) # Reshape to 2D array
        data_transformed = hand_feature_model.transform(data)
        pred_distance = hand_regression_model.predict(data_transformed)
        hand_distance = pred_distance[0]
        rounded_value = np.round(g.smoothed.face_pos.tuple[2], 2)
        clipped_value = np.clip(rounded_value, None, -1e-8)
        distance_scalar = clipped_value
        hand_distance = hand_distance / distance_scalar
        hand_distance += g.settings.hand_z_shifting
        hand_distance *= g.settings.hand_z_scalar
        hand_distance = np.interp(hand_distance, [-2, 2], [-1.2, 1])
        hand_distance = np.clip(hand_distance, -0.8, 0.0)
        hand_position[2] = hand_distance

        z = hand_pose[0] - hand_pose[17]
        x = np.cross(hand_pose[1] - hand_pose[0], z)
        y = np.cross(z, x)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        wrist_matrix = np.vstack((x, y, z)).T
        wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)

        finger_curl = finger_handling(hand_pose)
        finger_0, finger_1, finger_2, finger_3, finger_4 = finger_curl["thumb"],finger_curl["index"],finger_curl["middle"],finger_curl["ring"],finger_curl["pinky"]

        if hand_name == "Left":
            g.raw.left_hand.rotation.update(*wrist_rot)
            g.raw.left_hand.position.update(*hand_position)
            g.raw.left_hand.blendshapes.update(finger_0, finger_1, finger_2, finger_3, finger_4)
        else:
            g.raw.right_hand.rotation.update(*wrist_rot)
            g.raw.right_hand.position.update(*hand_position)
            g.raw.right_hand.blendshapes.update(finger_0, finger_1, finger_2, finger_3, finger_4)
    g.settings.handedness = handedness
