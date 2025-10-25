from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

import globals as g

def draw_hand_landmarks(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    landmarks = detection_result.multi_hand_landmarks
    handedness = detection_result.multi_handedness
    if landmarks is None or handedness is None: return rgb_image
    rgb_image = deepcopy(rgb_image)
    for hand, hand_landmarks in zip(handedness, landmarks):
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks.landmark  # Access through .landmark
        ])
        solutions.drawing_utils.draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )
        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = rgb_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            rgb_image,
            f"{'Right' if hand.classification[0].label == 'Left' else 'Left'}",  # Handedness label
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return rgb_image

def get_hand_pose(landmarks, reverse_flag=True):
    hand_pose = np.asarray([[l.x, l.y, l.z] for l in landmarks])
    if reverse_flag:
        hand_pose[:, 0] = -hand_pose[:, 0]  # flip the points a bit since steamvrs coordinate system is a bit diffrent
        hand_pose[:, 1] = -hand_pose[:, 1]
    return hand_pose

def calc_angle(vec1, vec2):
    with np.errstate(divide='ignore', invalid='ignore'):
        cross = np.linalg.norm(np.cross(vec1, vec2))
        dot = np.dot(vec1, vec2)
        return np.degrees(np.abs(np.arctan2(cross, dot)))

def finger_handling(hand_pose):
    global finger_mapper
    finger_curl= {}
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    for name in finger_names:
        cfg = g.config["Tracking"]["Finger"]
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
        # finger_curl[name] = finger_mapper[name]["mapper"](round(norm_value,1))
    return finger_curl

def compute_bounding_size(reference_kp):
    xs = [kp.x for kp in reference_kp]
    ys = [kp.y for kp in reference_kp]
    return max(xs)-min(xs), max(ys)-min(ys)

def calculate_normalized_distance(kp1, kp2, reference_kp, points):
    distances = [
        np.linalg.norm(np.array([kp1[i].x, kp1[i].y]) - np.array([kp2[i].x, kp2[i].y]))
        for i in points
    ]
    avg_distance = np.mean(distances)
    width, height = compute_bounding_size(reference_kp)
    return avg_distance / max(width, height)

prev_hands = {}  # {key: {'left': landmarks, 'right': landmarks}}
def hand_is_changed(key, hand_name, hand_landmarks, change_points, change_threshold, update_flag=True):
    global prev_hands
    all_keypoints = [hand_landmarks.landmark[i] for i in range(21)]
    reference_keypoints = [hand_landmarks.landmark[i] for i in [0, 1, 17]]
    other_hand = 'Right' if hand_name == 'Left' else 'Left'
    if key not in prev_hands:
        prev_hands[key] = {}
    swap_flag = False
    if prev_hands[key]:
        if hand_name in prev_hands[key] and other_hand in prev_hands[key]:
            self_dist = calculate_normalized_distance(
                all_keypoints,
                prev_hands[key][hand_name],
                reference_keypoints,
                change_points
            )
            other_dist = calculate_normalized_distance(
                all_keypoints,
                prev_hands[key][other_hand],
                reference_keypoints,
                change_points
            )
            swap_flag = other_dist < self_dist and (self_dist - other_dist) > g.config["Tracking"]["Hand"]["hand_swap_threshold"]
    changed = False
    norm_distance = 0
    if hand_name in prev_hands[key]:
        norm_distance = calculate_normalized_distance(
            all_keypoints,
            prev_hands[key][hand_name],
            reference_keypoints,
            change_points
        )
        changed = norm_distance >= change_threshold
    else:
        changed = True
    if update_flag and not swap_flag:
        prev_hands[key][hand_name] = all_keypoints
    return changed, norm_distance, swap_flag

hand_detection_counts = {"Left":0,"Right":0}
finger_action_threshold = {"Left":0,"Right":0}
prev_distance_scalar = None
def hand_pred_handling(backend, detection_result, hand_feature_model, hand_regression_model):
    global hand_detection_counts, finger_action_threshold,prev_distance_scalar

    g.hand_landmarks = detection_result.multi_hand_landmarks

    hand_detection_counts["Left"] -= 1
    if hand_detection_counts["Left"] < 0:
        hand_detection_counts["Left"] = 0
    hand_detection_counts["Right"] -= 1
    if hand_detection_counts["Right"] < 0:
        hand_detection_counts["Right"] = 0

    if detection_result.multi_hand_landmarks is not None and detection_result.multi_handedness is not None and detection_result.multi_hand_world_landmarks is not None:
        same_hand_flag=None
        if len(detection_result.multi_handedness)==2:
            hands=detection_result.multi_handedness
            if hands[0].classification[0].label == hands[1].classification[0].label:
                hand_name = "Right" if hands[0].classification[0].label == "Left" else "Left"
                _,avg_distance_0,_ = hand_is_changed("hand_change", hand_name, detection_result.multi_hand_landmarks[0],
                                                       [0,1,17],
                                                       0, False)
                _,avg_distance_1,_ = hand_is_changed("hand_change", hand_name, detection_result.multi_hand_landmarks[1],
                                                       [0,1,17],
                                                       0, False)
                if avg_distance_0<=avg_distance_1:
                    same_hand_flag=1
                else:
                    same_hand_flag=0

        for idx, (hand, hand_landmarks, hand_world_landmarks) in enumerate(
                zip(detection_result.multi_handedness, detection_result.multi_hand_landmarks,
                    detection_result.multi_hand_world_landmarks)):
            hand_name = "Right" if hand.classification[0].label == "Left" else "Left"
            if same_hand_flag == idx:
                continue
            else:
                _,_,_ = hand_is_changed("hand_change", hand_name, hand_landmarks,
                                                       [0,1,17],
                                                       0)
            if hand.classification[0].score < g.config["Tracking"]["Hand"]["hand_confidence"]:
                continue
            if hand_name == "Left":
                hand_detection_counts["Left"] += 2
                if hand_detection_counts["Left"] > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    hand_detection_counts["Left"] = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if hand_detection_counts["Left"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue
            else:
                hand_detection_counts["Right"] += 2
                if hand_detection_counts["Right"] > g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]:
                    hand_detection_counts["Right"] = g.config["Tracking"]["Hand"]["hand_detection_upper_threshold"]
                if hand_detection_counts["Right"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"]:
                    continue

            world_landmarks = hand_world_landmarks.landmark
            hand_pose = get_hand_pose(world_landmarks)
            image_landmarks = hand_landmarks.landmark
            image_hand_pose = get_hand_pose(image_landmarks, False)
            hand_position = g.smoothed.face_pos.array - image_hand_pose[9] # - image_hand_pose[2]
            hand_position[:2] *= [g.config["Tracking"]["Hand"]["x_scalar"], g.config["Tracking"]["Hand"]["y_scalar"]]
            # hand_distance_temp=np.linalg.norm(np.array(image_hand_pose[1][:2]) - np.array(image_hand_pose[2][:2]))
            # import keyboard
            # import pickle
            # if keyboard.is_pressed('a'):
            #     filename = f'./depth_dataset/image_hand_pose_{hand_name}_{index}.pkl'
            #     with open(filename, 'wb') as f:
            #         pickle.dump(image_hand_pose, f)
            #     print(f"image_hand_pose {filename}")
            #     index += 1

            keypoints = [5, 9, 13]
            data = image_hand_pose - image_hand_pose[0]
            data = data[keypoints].flatten()
            data = np.array(data)
            data = data.reshape(1, -1)  # Reshape to 2D array
            data_transformed = hand_feature_model.transform(data)
            pred_distance = hand_regression_model.predict(data_transformed)
            hand_distance_temp=pred_distance[0]

            rounded_value = np.round(g.data["HeadImagePosition"][2]["v"], 2)
            clipped_value = np.clip(rounded_value, None, -1e-8)
            distance_scalar = clipped_value

            hand_distance = hand_distance_temp/distance_scalar

            hand_distance += g.config["Tracking"]["Hand"]["z_shifting"]
            hand_distance *= g.config["Tracking"]["Hand"]["z_scalar"]
            hand_distance = np.interp(hand_distance, [-2, 2], [-1.2, 1])
            # print(hand_distance)
            if g.config["Tracking"]["Hand"]["only_front"]:
                hand_distance = np.clip(hand_distance, -0.8, 0.0)

            hand_position[2] = hand_distance

            position_change_flag,_,swap_flag = hand_is_changed("position",hand_name,hand_landmarks,g.config["Tracking"]["Hand"]["position_change_points"],g.config["Tracking"]["Hand"]["position_change_threshold"])
            if hand_name=="Left":
                backend.left_hand.change_flag=position_change_flag
            elif hand_name=="Right":
                backend.right_hand.change_flag=position_change_flag

            rotation_change_flag,_,_ = hand_is_changed("rotation",hand_name,hand_landmarks,g.config["Tracking"]["Hand"]["rotation_change_points"],g.config["Tracking"]["Hand"]["rotation_change_threshold"])
            z = hand_pose[0] - hand_pose[17]
            x = np.cross(hand_pose[1] - hand_pose[0], z)
            y = np.cross(z, x)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            z = z / np.linalg.norm(z)

            wrist_matrix = np.vstack((x, y, z)).T
            wrist_rot = R.from_matrix(wrist_matrix).as_euler("xyz", degrees=True)

            if g.config["Tracking"]["Finger"]["enable"]:
                finger_curl=finger_handling(hand_pose)
                finger_0, finger_1, finger_2, finger_3, finger_4 = finger_curl["thumb"],finger_curl["index"],finger_curl["middle"],finger_curl["ring"],finger_curl["pinky"]
                if g.config["Tracking"]["Hand"]["enable_finger_action"]:
                    if finger_1 < g.config["Tracking"]["Hand"]["trigger_threshold"]:
                        backend.send_trigger(True if hand_name=="Left" else False, 0, 1)
                    else:
                        backend.send_trigger(True if hand_name=="Left" else False, 0, 0)
                    if finger_1>0.3 and finger_3>0.3 and finger_4 >0.5 and finger_0<0.7 and finger_2<0.4:
                        finger_action_threshold[hand_name] = g.config["Tracking"]["Hand"]["finger_action_threshold"]
                    else:
                        finger_action_threshold[hand_name] = max(0,finger_action_threshold[hand_name]-1)
                    if finger_action_threshold[hand_name] != 0:
                        finger_1 = 1.0
                        finger_3 = 1.0
                        finger_4 = 1.0
                        finger_0 = 0.0
                        finger_2 = 0.25
            else:
                finger_0, finger_1, finger_2, finger_3, finger_4 = 1.0, 1.0, 1.0, 1.0, 1.0

            if hand_name == "Left":
                if rotation_change_flag:
                    g.raw.left_hand.rotation.update(*wrist_rot)
                if position_change_flag:
                    g.raw.left_hand.position.update(*hand_position)
                g.raw.left_hand.blendshapes.update(finger_0, finger_1, finger_2, finger_3, finger_4)
                backend.left_hand.enable = True
            else:
                if rotation_change_flag:
                    g.raw.right_hand.rotation.update(*wrist_rot)
                if position_change_flag:
                    g.raw.right_hand.position.update(*hand_position)
                g.raw.right_hand.blendshapes.update(finger_0, finger_1, finger_2, finger_3, finger_4)
                backend.right_hand.enable = True

    if hand_detection_counts["Left"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"]:
        backend.left_hand.enable = False

    if hand_detection_counts["Right"] <= g.config["Tracking"]["Hand"]["hand_detection_lower_threshold"] and g.config["Tracking"]["Hand"]["enable_hand_auto_reset"]:
        backend.right_hand.enable = False
