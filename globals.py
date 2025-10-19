import json
from copy import deepcopy

only_ingame = False
only_ingame_game = ''

config = json.load(open('./settings/config.json'))
data = json.load(open('./settings/data.json'))
default_data = deepcopy(data)
latest_data = [0.0] * (64 + 6 + 12 + 10 + 12 + 10 + 3 + 2)

smoothing_enabled = True
gesture_config = json.load(open('./settings/gestures.json'))
face_landmarks = None
hand_landmarks = None
handedness = None
pose_landmarks = None
