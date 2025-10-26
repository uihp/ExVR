import globals as g
from scipy.spatial.transform import Rotation as R
import numpy as np

from globals import OSC
from classes import *

class GloveControllerSender:
    def __init__(self):
        self.pointer_mode = True
        self.left_hand = TrackingUnit(Position(0,0,0), Quaternion(0,0,0,1), BlendShapeGroup([0,0,0,0,0]))
        self.right_hand = TrackingUnit(Position(0,0,0), Quaternion(0,0,0,1), BlendShapeGroup([0,0,0,0,0]))
        OSC.send_message('/VMT/SetRoomMatrix', [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0])
    def update(self):
        eular = (g.smoothed.left_hand.rotation.array if not self.pointer_mode else np.array([-20, 0, -15]))
        eular += [75, 0, -15]
        quat = R.from_euler("xyz", eular, degrees=True).as_quat()
        matrix = R.from_euler("xyz", eular, degrees=True).as_matrix()
        center = np.array([0.05, -0.07, -0.12])
        self.left_hand.position.update(*(matrix @ center + (g.smoothed.left_hand.position.array if not self.pointer_mode else np.array([-0.13, -0.05, -0.25]))))
        self.left_hand.rotation.update(*quat)
        self.left_hand.blendshapes.update(*g.smoothed.left_hand.blendshapes.tuple)

        eular = g.smoothed.right_hand.rotation.array
        eular += [75, 0, -15]
        quat = R.from_euler("xyz", eular, degrees=True).as_quat()
        matrix = R.from_euler("xyz", eular, degrees=True).as_matrix()
        center = np.array([-0.05, -0.07, -0.12])
        self.right_hand.position.update(*(matrix @ center + g.smoothed.right_hand.position.array))
        self.right_hand.rotation.update(*quat)
        self.right_hand.blendshapes.update(*g.smoothed.right_hand.blendshapes.tuple)

        if g.handedness[0] or self.pointer_mode:
            OSC.send_message("/VMT/Joint/Driver", [1, 5, 0.0, *self.left_hand.position.tuple, *self.left_hand.rotation.tuple, 'HMD'])
            for i, e in enumerate(self.left_hand.blendshapes.tuple):
                OSC.send_message("/VMT/Skeleton/Scalar", [1, i+1, e, 0, 0])
            OSC.send_message("/VMT/Skeleton/Apply", [1, 0.0])
        else: OSC.send_message("/VMT/Room/Unity", [1, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if g.handedness[1]:
            OSC.send_message("/VMT/Joint/Driver", [2, 6, 0.0, *self.right_hand.position.tuple, *self.right_hand.rotation.tuple, 'HMD'])
            for i, e in enumerate(self.right_hand.blendshapes.tuple):
                OSC.send_message("/VMT/Skeleton/Scalar", [2, i+1, e, 0, 0])
            OSC.send_message("/VMT/Skeleton/Apply", [2, 0.0])
        else: OSC.send_message("/VMT/Room/Unity", [2, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
