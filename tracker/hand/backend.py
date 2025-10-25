from dataclasses import dataclass
from pythonosc import udp_client
import globals as g
from scipy.spatial.transform import Rotation as R
import numpy as np

from classes import *

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
    change_flag: bool

# GloveControllerSender equivalent in Python
class GloveControllerSender:
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 39570):
        # Initialize OSC client
        self.client = udp_client.SimpleUDPClient(osc_ip, osc_port)

        self.left_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0), False, True)
        self.right_hand = Transform((0, 0, 0), (0, 0, 0, 1), (1.0, 1.0, 1.0, 1.0, 1.0), False, True)
        self.vmt_init()

    def send_hand(self, is_left_hand, target: Transform):
        from pprint import pp
        #pp(g.smoothed.left_hand.rotation.tuple) # (-59.83069652593504, 77.7619216071455, -130.3005959491607)
        message = [
            1 if is_left_hand else 2,  # lefthand ? 1 : 2
            5 if is_left_hand else 6,  # enable
            0.0,  # timeoffset
            *target.position,
            *target.rotation,
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

    def update(self):
        eular = g.smoothed.left_hand.rotation.array
        eular += [75, 0, -15]
        quat = R.from_euler("xyz", eular, degrees=True).as_quat()
        matrix = R.from_euler("xyz", eular, degrees=True).as_matrix()
        center = np.array([0.05, -0.07, -0.12])
        self.left_hand.position = matrix @ center + g.smoothed.left_hand.position.array
        self.left_hand.rotation = quat
        self.left_hand.finger = g.smoothed.left_hand.blendshapes.tuple

        eular = g.smoothed.right_hand.rotation.array
        eular += [75, 0, -15]
        quat = R.from_euler("xyz", eular, degrees=True).as_quat()
        matrix = R.from_euler("xyz", eular, degrees=True).as_matrix()
        center = np.array([-0.05, -0.07, -0.12])
        self.right_hand.position = matrix @ center + g.smoothed.right_hand.position.array
        self.right_hand.rotation = quat
        self.right_hand.finger = g.smoothed.right_hand.blendshapes.tuple

        if not self.left_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"]:
            self.send_trigger(True, 0, 0)
            self.disable_hand(True)
        else:
            self.send_hand(True, self.left_hand)
            self.send_finger(True, self.left_hand)

        if not self.right_hand.enable and g.config["Tracking"]["Hand"]["enable_hand_down"]:
            self.send_trigger(False, 0, 0)
            self.disable_hand(False)
        else:
            self.send_hand(False, self.right_hand)
            self.send_finger(False, self.right_hand)
