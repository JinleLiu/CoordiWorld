"""Coordinate transform helpers for ICA-style SceneSummary generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class RadarEgoBev:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True)
class ImageProjection:
    u: float
    v: float
    depth: float


def transform_point_sensor_to_ego(
    x: float,
    y: float,
    sensor_to_ego: Matrix | None = None,
) -> tuple[float, float]:
    """Transform a 2D sensor-frame point into ego-BEV coordinates."""
    if sensor_to_ego is None:
        return float(x), float(y)

    rows = _as_matrix(sensor_to_ego)
    if len(rows) == 3 and all(len(row) == 3 for row in rows):
        ego_x = rows[0][0] * x + rows[0][1] * y + rows[0][2]
        ego_y = rows[1][0] * x + rows[1][1] * y + rows[1][2]
        return ego_x, ego_y
    if len(rows) == 4 and all(len(row) == 4 for row in rows):
        ego_x = rows[0][0] * x + rows[0][1] * y + rows[0][3]
        ego_y = rows[1][0] * x + rows[1][1] * y + rows[1][3]
        return ego_x, ego_y
    raise ValueError("sensor_to_ego must be a 3x3 or 4x4 matrix")


def transform_yaw_sensor_to_ego(yaw: float, sensor_to_ego: Matrix | None = None) -> float:
    """Rotate a planar yaw angle from sensor frame into ego frame."""
    if sensor_to_ego is None:
        return float(yaw)
    rows = _as_matrix(sensor_to_ego)
    if len(rows) not in {3, 4} or any(len(row) != len(rows) for row in rows):
        raise ValueError("sensor_to_ego must be a 3x3 or 4x4 matrix")
    rotation_yaw = math.atan2(rows[1][0], rows[0][0])
    return _wrap_angle(float(yaw) + rotation_yaw)


def transform_velocity_sensor_to_ego(
    vx: float,
    vy: float,
    sensor_to_ego: Matrix | None = None,
) -> tuple[float, float]:
    """Rotate a planar velocity vector from sensor frame into ego frame."""
    if sensor_to_ego is None:
        return float(vx), float(vy)
    rows = _as_matrix(sensor_to_ego)
    if len(rows) not in {3, 4} or any(len(row) != len(rows) for row in rows):
        raise ValueError("sensor_to_ego must be a 3x3 or 4x4 matrix")
    ego_vx = rows[0][0] * vx + rows[0][1] * vy
    ego_vy = rows[1][0] * vx + rows[1][1] * vy
    return ego_vx, ego_vy


def radar_polar_to_ego_bev(
    range_m: float,
    azimuth_rad: float,
    radial_velocity_mps: float = 0.0,
    sensor_to_ego: Matrix | None = None,
) -> RadarEgoBev:
    """Convert a radar polar point and radial velocity into ego-BEV coordinates."""
    sensor_x = float(range_m) * math.cos(float(azimuth_rad))
    sensor_y = float(range_m) * math.sin(float(azimuth_rad))
    sensor_vx = float(radial_velocity_mps) * math.cos(float(azimuth_rad))
    sensor_vy = float(radial_velocity_mps) * math.sin(float(azimuth_rad))

    ego_x, ego_y = transform_point_sensor_to_ego(sensor_x, sensor_y, sensor_to_ego)
    ego_vx, ego_vy = transform_velocity_sensor_to_ego(sensor_vx, sensor_vy, sensor_to_ego)
    return RadarEgoBev(x=ego_x, y=ego_y, vx=ego_vx, vy=ego_vy)


def project_ego_point_to_camera(
    x: float,
    y: float,
    z: float,
    camera_from_ego: Matrix,
    intrinsic: Matrix,
) -> ImageProjection | None:
    """Project an ego-frame 3D point into the camera image plane."""
    extrinsic = _as_matrix(camera_from_ego)
    intr = _as_matrix(intrinsic)
    if len(extrinsic) != 4 or any(len(row) != 4 for row in extrinsic):
        raise ValueError("camera_from_ego must be a 4x4 matrix")
    if len(intr) != 3 or any(len(row) != 3 for row in intr):
        raise ValueError("intrinsic must be a 3x3 matrix")

    cam_x = extrinsic[0][0] * x + extrinsic[0][1] * y + extrinsic[0][2] * z + extrinsic[0][3]
    cam_y = extrinsic[1][0] * x + extrinsic[1][1] * y + extrinsic[1][2] * z + extrinsic[1][3]
    cam_z = extrinsic[2][0] * x + extrinsic[2][1] * y + extrinsic[2][2] * z + extrinsic[2][3]
    if cam_z <= 0:
        return None

    img_x = intr[0][0] * cam_x + intr[0][1] * cam_y + intr[0][2] * cam_z
    img_y = intr[1][0] * cam_x + intr[1][1] * cam_y + intr[1][2] * cam_z
    img_z = intr[2][0] * cam_x + intr[2][1] * cam_y + intr[2][2] * cam_z
    if img_z == 0:
        return None
    return ImageProjection(u=img_x / img_z, v=img_y / img_z, depth=cam_z)


def point_in_bbox(u: float, v: float, bbox_xyxy: Sequence[float]) -> bool:
    """Return whether an image point lies inside [x1, y1, x2, y2]."""
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return x1 <= float(u) <= x2 and y1 <= float(v) <= y2


def _as_matrix(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
