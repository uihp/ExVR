from pythonosc import udp_client
from classes import *

OSC = udp_client.SimpleUDPClient('127.0.0.1', 39570)

class settings:
    camera_url = 'wss://192.168.31.150:5000/browsercam/signal'
    flip_x = False
    flip_y = False
    priority = 'ABOVE_NORMAL_PRIORITY_CLASS'
    only_ingame = False
    only_ingame_game = ''
    handedness = [False, False]
    # scalars & shiftings
    x_scalar = 1
    y_scalar = 1
    z_scalar = 1
    yaw_scalar = 1
    pitch_scalar = 1
    roll_scalar = 1
    hand_x_scalar = 1.1
    hand_y_scalar = 1.0
    hand_z_scalar = 0.55
    hand_z_shifting = 1.4472277290670614
    @classmethod
    def toggle(cls, key): setattr(cls, key, not getattr(cls, key))

class raw:
    head = TrackingUnit(Position(0,0,50), Rotation(0,0,0))
    face_pos = Position(0,0,0)
    mouse = Rotation(0,0,0)
    left_hand = TrackingUnit(Position(0,0,0), Rotation(0,0,0), BlendShapeGroup([0,0,0,0,0]))
    right_hand = TrackingUnit(Position(0,0,0), Rotation(0,0,0), BlendShapeGroup([0,0,0,0,0]))

class smoothed:
    head = TrackingUnit(Position(0,0,0), Rotation(0,0,0))
    face_pos = Position(0,0,0)
    left_hand = TrackingUnit(Position(0,0,0), Rotation(0,0,0), BlendShapeGroup([0,0,0,0,0]))
    right_hand = TrackingUnit(Position(0,0,0), Rotation(0,0,0), BlendShapeGroup([0,0,0,0,0]))
