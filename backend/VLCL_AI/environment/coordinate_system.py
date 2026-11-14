import numpy as np
from typing import Tuple, List, Union
from loguru import logger

def infer_room_bounds(room_dims, led_positions) -> Tuple[float, float, float, float, float, float]:
    """
    Room extent (min_x, max_x, min_y, max_y, min_z, max_z) for a [W, L, H] room.

    Two layouts are in use: the room frame with the origin at a floor corner
    (0 <= x <= W), and the paper's layout centred on the origin (LEDs at +/-0.4 m).
    An LED with a negative x or y means the centred layout.
    """
    W, L, H = (float(v) for v in room_dims)
    pts = list(led_positions.values()) if isinstance(led_positions, dict) else list(led_positions)
    if any(p[0] < 0 or p[1] < 0 for p in pts):
        return (-W / 2.0, W / 2.0, -L / 2.0, L / 2.0, 0.0, H)
    return (0.0, W, 0.0, L, 0.0, H)


class CoordinateSystem:
    """
    Manages coordinate transformations, reference frames, and rotations (Roll, Pitch, Yaw)
    relative to the Room Frame where the origin is at the bottom-left floor corner (0,0,0).
    """
    
    @staticmethod
    def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Computes the rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll) using radians.
        Using extrinsic/intrinsic ZYX standard convention.
        """
        # Convert degrees to radians
        r = np.radians(roll)
        p = np.radians(pitch)
        y = np.radians(yaw)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])
        
        Ry = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])
        
        # R = Rz * Ry * Rx
        return Rz @ (Ry @ Rx)

    @staticmethod
    def transform_to_local(point_global: np.ndarray, origin_global: np.ndarray, R_global_to_local: np.ndarray) -> np.ndarray:
        """Transforms a 3D point from Global room coordinates to Local receiver coordinates."""
        diff = point_global - origin_global
        return R_global_to_local.T @ diff

    @staticmethod
    def transform_to_global(point_local: np.ndarray, origin_global: np.ndarray, R_global_to_local: np.ndarray) -> np.ndarray:
        """Transforms a 3D point from Local receiver coordinates to Global room coordinates."""
        return origin_global + (R_global_to_local @ point_local)

    @staticmethod
    def normalize_vector(vec: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Returns the normalized vector."""
        v = np.array(vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def vector_to_angles(vec: np.ndarray) -> Tuple[float, float]:
        """
        Calculates azimuth and elevation angles of a vector in degrees.
        Azimuth: Angle on X-Y plane from X axis [0, 360).
        Elevation: Angle from X-Y plane [-90, 90].
        """
        v = CoordinateSystem.normalize_vector(vec)
        x, y, z = v[0], v[1], v[2]
        
        elevation = np.degrees(np.arcsin(z))
        azimuth = np.degrees(np.arctan2(y, x)) % 360.0
        
        return azimuth, elevation

    @staticmethod
    def angles_to_vector(azimuth: float, elevation: float) -> np.ndarray:
        """Converts azimuth and elevation (in degrees) to a unit direction vector."""
        az = np.radians(azimuth)
        el = np.radians(elevation)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        return np.array([x, y, z])
