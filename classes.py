from dataclasses import dataclass
import numpy as np

@dataclass
class Position:
    x: float
    y: float
    z: float
    @property
    def tuple(self): return (self.x, self.y, self.z)
    @property
    def array(self): return np.array([self.x, self.y, self.z])
    def update(self, x, y, z): self.x, self.y, self.z = x, y, z

@dataclass
class Rotation:
    yaw: float
    pitch: float
    roll: float
    @property
    def tuple(self): return (self.yaw, self.pitch, self.roll)
    @property
    def array(self): return np.array([self.yaw, self.pitch, self.roll])
    def update(self, yaw, pitch, roll): self.yaw, self.pitch, self.roll = yaw % 360, pitch % 360, roll % 360

@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float
    @property
    def tuple(self): return (self.x, self.y, self.z, self.w)
    @property
    def array(self): return np.array([self.x, self.y, self.z, self.w])
    def update(self, x, y, z, w): self.x, self.y, self.z, self.w = x, y, z, w

@dataclass
class BlendShapeGroup:
    weights: list[float]
    @property
    def tuple(self): return tuple(self.weights)
    @property
    def array(self): return np.array(self.weights)
    def update(self, *W): self.weights = W

@dataclass
class TrackingUnit:
    position: Position
    rotation: Rotation | Quaternion
    blendshapes: BlendShapeGroup = None
