from dataclasses import dataclass
from pythonosc import udp_client
import globals as g
from scipy.spatial.transform import Rotation as R
import numpy as np

def get_value(value, value_d):
    if value["e"]:
        return value["v"] + value["s"]
    else:
        return value_d["v"] + value_d["s"]

def calculate_endpoint(start_point, length, euler_angles):
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    direction_vector = np.array([0, 0, -length])
    rotated_vector = rotation.apply(direction_vector)
    endpoint = np.array(start_point) + rotated_vector
    return endpoint

# Define a simple Transform class to hold position and rotation g.data
@dataclass
class Transform:
    position: tuple  # (x, y, z)
    rotation: tuple  # (x, y, z, w)
    finger: tuple  # (0,1,2,3,4)
    enable: bool
    force_enable: bool
    change_flag: bool

# GloveControllerSender equivalent in Python
class GloveControllerSender:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 39570):
        # Initialize OSC client
        self.client = udp_client.SimpleUDPClient(osc_ip, osc_port)

        self.left_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0), False, False, True)
        self.right_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0), False, False, True)
        self.vmt_init()

    def send_hand(self, is_left_hand, target: Transform):
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            5 if is_left_hand else 6,  # enable
            0.0,  # timeoffset
            target.position[0],
            target.position[1],
            target.position[2],  # -0.25,
            target.rotation[0],
            target.rotation[1],
            target.rotation[2],
            target.rotation[3],
            "HMD",  # serial
        ]
        self.client.send_message("/VMT/Joint/Driver", message)

    def disable_hand(self,is_left_hand):
        message = [1 if is_left_hand else 2, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.client.send_message("/VMT/Room/Unity", message)

    def send_finger(self, is_left_hand, target: Transform):
        for i, value in enumerate(target.finger):
            index = i + 1
            message_0 = [
                1 if is_left_hand else 2,  # lefthand ? 1 : 2
                int(index),
                float(value),
                0,
                0,
            ]
            self.client.send_message("/VMT/Skeleton/Scalar", message_0)
        message_1 = [1 if is_left_hand else 2, 0.0]  # lefthand ? 1 : 2
        self.client.send_message("/VMT/Skeleton/Apply", message_1)

    def send_trigger(self, is_left_hand, index, status=0.0):
        message = [1 if is_left_hand else 2, int(index), 0.0, float(status)]
        self.client.send_message("/VMT/Input/Trigger", message)

    def vmt_init(self):
        self.client.send_message(
            "/VMT/SetRoomMatrix",
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0],
        )

    def handling_hand_data(self):
        if g.config["Tracking"]["LeftController"]["enable"]:
            left_hand_type="Controller"
        else:
            left_hand_type = "Hand"
        if g.config["Tracking"]["RightController"]["enable"]:
            right_hand_type="Controller"
        else:
            right_hand_type="Hand"

        # Process left hand g.data
        yaw_l = get_value(g.data[f"Left{left_hand_type}Rotation"][0], g.default_data[f"Left{left_hand_type}Rotation"][0])
        pitch_l = get_value(
            g.data[f"Left{left_hand_type}Rotation"][1], g.default_data[f"Left{left_hand_type}Rotation"][1]
        )
        roll_l = get_value(g.data[f"Left{left_hand_type}Rotation"][2], g.default_data[f"Left{left_hand_type}Rotation"][2])
        if g.config["Tracking"]["LeftController"]["enable"]:
            base_x_l=g.config["Tracking"]["LeftController"]["base_x"]
            base_y_l=g.config["Tracking"]["LeftController"]["base_y"]
            base_z_l=g.config["Tracking"]["LeftController"]["base_z"]
            length_l=g.config["Tracking"]["LeftController"]["length"]
            g.data[f"Left{left_hand_type}Position"][0]["v"],g.data[f"Left{left_hand_type}Position"][1]["v"],g.data[f"Left{left_hand_type}Position"][2]["v"] = calculate_endpoint([base_x_l,base_y_l,base_z_l], length_l, [yaw_l-40,pitch_l,roll_l])

        x_l = get_value(g.data[f"Left{left_hand_type}Position"][0], g.default_data[f"Left{left_hand_type}Position"][0])
        y_l = get_value(g.data[f"Left{left_hand_type}Position"][1], g.default_data[f"Left{left_hand_type}Position"][1])
        z_l = get_value(g.data[f"Left{left_hand_type}Position"][2], g.default_data[f"Left{left_hand_type}Position"][2])
        quat_l = R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_quat()
        matrix_l=R.from_euler("xyz", [yaw_l, pitch_l, roll_l], degrees=True).as_matrix()
        # Process right hand g.data
        yaw_r = get_value(
            g.data[f"Right{right_hand_type}Rotation"][0], g.default_data[f"Right{right_hand_type}Rotation"][0]
        )
        pitch_r = get_value(
            g.data[f"Right{right_hand_type}Rotation"][1], g.default_data[f"Right{right_hand_type}Rotation"][1]
        )
        roll_r = get_value(
            g.data[f"Right{right_hand_type}Rotation"][2], g.default_data[f"Right{right_hand_type}Rotation"][2]
        )
        if g.config["Tracking"]["RightController"]["enable"]:
            base_x_r=g.config["Tracking"]["RightController"]["base_x"]
            base_y_r=g.config["Tracking"]["RightController"]["base_y"]
            base_z_r=g.config["Tracking"]["RightController"]["base_z"]
            length_r=g.config["Tracking"]["RightController"]["length"]
            g.data[f"Right{right_hand_type}Position"][0]["v"],g.data[f"Right{right_hand_type}Position"][1]["v"],g.data[f"Right{right_hand_type}Position"][2]["v"] = calculate_endpoint([base_x_r,base_y_r,base_z_r], length_r, [yaw_r-40,pitch_r,roll_r])

        x_r = get_value(g.data[f"Right{right_hand_type}Position"][0], g.default_data[f"Right{right_hand_type}Position"][0])
        y_r = get_value(g.data[f"Right{right_hand_type}Position"][1], g.default_data[f"Right{right_hand_type}Position"][1])
        z_r = get_value(g.data[f"Right{right_hand_type}Position"][2], g.default_data[f"Right{right_hand_type}Position"][2])
        quat_r = R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_quat()
        matrix_r=R.from_euler("xyz", [yaw_r, pitch_r, roll_r], degrees=True).as_matrix()

        if not g.config["Tracking"]["Pose"]["enable"]:
            center_l = np.array([g.config["Tracking"]["Hand"]["center_l_x"], g.config["Tracking"]["Hand"]["center_l_y"],
                            g.config["Tracking"]["Hand"]["center_l_z"]])
        else:
            center_l = np.array([g.config["Tracking"]["Pose"]["center_l_x"], g.config["Tracking"]["Pose"]["center_l_y"],
                            g.config["Tracking"]["Pose"]["center_l_z"]])

        wrist_position_l = (x_l, y_l, z_l)
        wrist_position_l = np.array(wrist_position_l)
        wrist_position_l = matrix_l @ center_l + wrist_position_l
        wrist_position_l=(wrist_position_l[0], wrist_position_l[1] ,wrist_position_l[2])
        self.left_hand.position = wrist_position_l
        self.left_hand.rotation = quat_l

        if not g.config["Tracking"]["Pose"]["enable"]:
            center_r = np.array([g.config["Tracking"]["Hand"]["center_r_x"], g.config["Tracking"]["Hand"]["center_r_y"],
                            g.config["Tracking"]["Hand"]["center_r_z"]])
        else:
            center_r = np.array([g.config["Tracking"]["Pose"]["center_r_x"], g.config["Tracking"]["Pose"]["center_r_y"],
                            g.config["Tracking"]["Pose"]["center_r_z"]])

        wrist_position_r = (x_r, y_r, z_r)
        wrist_position_r = np.array(wrist_position_r)
        wrist_position_r = matrix_r @ center_r + wrist_position_r
        wrist_position_r=(wrist_position_r[0], wrist_position_r[1] ,wrist_position_r[2])
        self.right_hand.position = wrist_position_r
        self.right_hand.rotation = quat_r

        finger_l = tuple(
            get_value(v, v_d)
            for v, v_d in zip(g.data[f"Left{left_hand_type}Finger"], g.default_data[f"Left{left_hand_type}Finger"])
        )
        finger_r = tuple(
            get_value(v, v_d)
            for v, v_d in zip(g.data[f"Right{right_hand_type}Finger"], g.default_data[f"Right{right_hand_type}Finger"])
        )
        # print(f"Right{right_hand_type}Finger",finger_r)

        self.left_hand.finger = finger_l
        self.right_hand.finger = finger_r

    def update(self):
        self.handling_hand_data()
        if not self.left_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"] and not self.left_hand.force_enable:
            self.send_trigger(True, 0, 0)
            self.disable_hand(True)
        else:
            self.send_hand(True, self.left_hand)
            self.send_finger(True, self.left_hand)

        if not self.right_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"] and not self.right_hand.force_enable:
            self.send_trigger(False, 0, 0)
            self.disable_hand(False)
        else:
            self.send_hand(False, self.right_hand)
            self.send_finger(False, self.right_hand)
