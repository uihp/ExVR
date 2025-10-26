import socket
import struct

import globals as g

class TrackerBackend:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def update(self):
        # print(' '*100, end='\r')
        # print(*map(int, (*g.smoothed.head.position.tuple, *g.smoothed.head.rotation.tuple, *g.raw.mouse.tuple)), sep='\t', end='\r')
        packed_hmd_data = struct.pack('9d', *g.smoothed.head.position.tuple, *g.smoothed.head.rotation.tuple, *g.raw.mouse.tuple)
        self.socket.sendto(packed_hmd_data, ('127.0.0.1', 4242))
