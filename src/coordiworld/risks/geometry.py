"""Geometry primitives for CoordiWorld risk heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

Point2D = tuple[float, float]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class Box2D:
    x: float
    y: float
    yaw: float
    length: float
    width: float


@dataclass(frozen=True)
class BoxInteraction:
    center_distance: float
    clearance: float
    overlap: bool
    soft_collision_probability: float


def pose_from_sequence(values: Sequence[float]) -> Pose2D:
    if len(values) < 2:
        raise ValueError("pose must contain at least [x, y]")
    yaw = float(values[2]) if len(values) >= 3 else 0.0
    return Pose2D(x=_number(values[0]), y=_number(values[1]), yaw=yaw)


def box_from_pose(
    pose: Sequence[float] | Pose2D,
    *,
    length: float,
    width: float,
) -> Box2D:
    if length <= 0 or width <= 0:
        raise ValueError("box length and width must be > 0")
    resolved_pose = pose if isinstance(pose, Pose2D) else pose_from_sequence(pose)
    return Box2D(
        x=resolved_pose.x,
        y=resolved_pose.y,
        yaw=resolved_pose.yaw,
        length=float(length),
        width=float(width),
    )


def oriented_box_corners(box: Box2D) -> list[Point2D]:
    half_length = box.length / 2.0
    half_width = box.width / 2.0
    local = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    cos_yaw = math.cos(box.yaw)
    sin_yaw = math.sin(box.yaw)
    return [
        (
            box.x + local_x * cos_yaw - local_y * sin_yaw,
            box.y + local_x * sin_yaw + local_y * cos_yaw,
        )
        for local_x, local_y in local
    ]


def box_interaction_feature(
    ego_box: Box2D,
    agent_box: Box2D,
    *,
    margin: float = 0.0,
    sigmoid_scale: float = 0.75,
) -> BoxInteraction:
    distance = math.hypot(ego_box.x - agent_box.x, ego_box.y - agent_box.y)
    ego_radius = 0.5 * math.hypot(ego_box.length, ego_box.width)
    agent_radius = 0.5 * math.hypot(agent_box.length, agent_box.width)
    clearance = distance - ego_radius - agent_radius - margin
    overlap = polygons_overlap(oriented_box_corners(ego_box), oriented_box_corners(agent_box))
    probability = sigmoid_soft_margin(-clearance, scale=sigmoid_scale)
    if overlap:
        probability = max(probability, 0.99)
    return BoxInteraction(
        center_distance=distance,
        clearance=clearance,
        overlap=overlap,
        soft_collision_probability=clip01(probability),
    )


def polygons_overlap(first: Sequence[Point2D], second: Sequence[Point2D]) -> bool:
    """Check convex polygon overlap with separating axis theorem."""
    for polygon in (first, second):
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            edge_x = next_point[0] - point[0]
            edge_y = next_point[1] - point[1]
            axis = (-edge_y, edge_x)
            first_min, first_max = _project_polygon(first, axis)
            second_min, second_max = _project_polygon(second, axis)
            if first_max < second_min or second_max < first_min:
                return False
    return True


def point_in_polygon(point: Point2D, polygon: Sequence[Sequence[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous_x = _number(polygon[-1][0])
    previous_y = _number(polygon[-1][1])
    for raw_point in polygon:
        current_x = _number(raw_point[0])
        current_y = _number(raw_point[1])
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y)
            intersection_x /= previous_y - current_y
            intersection_x += current_x
            if x < intersection_x:
                inside = not inside
        previous_x = current_x
        previous_y = current_y
    return inside


def distance_point_to_polyline(point: Point2D, polyline: Sequence[Sequence[float]]) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return math.hypot(point[0] - _number(polyline[0][0]), point[1] - _number(polyline[0][1]))
    return min(
        distance_point_to_segment(point, _as_point(polyline[index]), _as_point(polyline[index + 1]))
        for index in range(len(polyline) - 1)
    )


def distance_point_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def trajectory_crosses_polyline(
    previous_pose: Sequence[float],
    current_pose: Sequence[float],
    polyline: Sequence[Sequence[float]],
) -> bool:
    if len(polyline) < 2:
        return False
    start = (_number(previous_pose[0]), _number(previous_pose[1]))
    end = (_number(current_pose[0]), _number(current_pose[1]))
    return any(
        segments_intersect(start, end, _as_point(polyline[index]), _as_point(polyline[index + 1]))
        for index in range(len(polyline) - 1)
    )


def segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    def orientation(p: Point2D, q: Point2D, r: Point2D) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    epsilon = 1e-9
    if abs(o1) < epsilon and on_segment(a, c, b):
        return True
    if abs(o2) < epsilon and on_segment(a, d, b):
        return True
    if abs(o3) < epsilon and on_segment(c, a, d):
        return True
    if abs(o4) < epsilon and on_segment(c, b, d):
        return True
    return False


def sigmoid_soft_margin(value: float, *, scale: float = 1.0) -> float:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    normalized = max(-60.0, min(60.0, float(value) / scale))
    return 1.0 / (1.0 + math.exp(-normalized))


def smooth_max(values: Sequence[float], *, temperature: float = 0.2) -> float:
    if not values:
        return 0.0
    if temperature <= 0:
        return max(values)
    maximum = max(values)
    smoothed = maximum + temperature * math.log(
        sum(math.exp((value - maximum) / temperature) for value in values)
    )
    return clip01(smoothed)


def noisy_or(probabilities: Sequence[float]) -> float:
    survival = 1.0
    for probability in probabilities:
        survival *= 1.0 - clip01(probability)
    return clip01(1.0 - survival)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _project_polygon(polygon: Sequence[Point2D], axis: Point2D) -> tuple[float, float]:
    axis_x, axis_y = axis
    norm = math.hypot(axis_x, axis_y)
    if norm == 0:
        raise ValueError("polygon edge axis must be non-zero")
    unit_x = axis_x / norm
    unit_y = axis_y / norm
    projections = [point[0] * unit_x + point[1] * unit_y for point in polygon]
    return min(projections), max(projections)


def _as_point(raw: Sequence[float]) -> Point2D:
    if len(raw) != 2:
        raise ValueError("point must be [x, y]")
    return _number(raw[0]), _number(raw[1])


def _number(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("geometry values must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError("geometry values must be finite")
    return float(value)
