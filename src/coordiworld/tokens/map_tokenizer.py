"""Map tokenization helpers for structured SceneSummary map tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from coordiworld.scene_summary.schema import EgoState, MapToken

MAP_TYPE_CODES: dict[str, float] = {
    "lane_centerline": 1.0,
    "drivable_area": 2.0,
    "stop_line": 3.0,
    "traffic_light": 4.0,
    "conflict_zone": 5.0,
}
TRAFFIC_STATE_CODES: dict[str | None, float] = {
    None: 0.0,
    "unknown": 0.0,
    "red": 1.0,
    "yellow": 2.0,
    "green": 3.0,
}
MAP_RISK_PRIOR: dict[str, float] = {
    "conflict_zone": 3.0,
    "traffic_light": 2.5,
    "stop_line": 2.0,
    "lane_centerline": 0.5,
    "drivable_area": 0.25,
}
MAP_FEATURE_DIM = 10


@dataclass(frozen=True)
class TokenizedMap:
    map_tensor: list[list[float]]
    mask: list[int]
    selected_ids: list[str]


@dataclass(frozen=True)
class MapTokenizerConfig:
    max_map_tokens: int = 24


class MapTokenizer:
    """Convert SceneSummary map tokens into padded numeric features."""

    def __init__(self, config: MapTokenizerConfig | None = None) -> None:
        self.config = config or MapTokenizerConfig()
        if self.config.max_map_tokens <= 0:
            raise ValueError("max_map_tokens must be > 0")

    def tokenize(self, map_tokens: Sequence[MapToken], ego: EgoState) -> TokenizedMap:
        selected = select_map_tokens(map_tokens, ego, self.config.max_map_tokens)
        tensor = [_encode_map_token(token, ego) for token in selected]
        mask = [1] * len(tensor)

        while len(tensor) < self.config.max_map_tokens:
            tensor.append([0.0] * MAP_FEATURE_DIM)
            mask.append(0)

        return TokenizedMap(
            map_tensor=tensor,
            mask=mask,
            selected_ids=[token.id for token in selected],
        )


def map_feature_order() -> list[str]:
    return [
        "rel_center_x",
        "rel_center_y",
        "center_distance",
        "point_count",
        "geometry_span",
        "type_code",
        "traffic_state_code",
        "has_polyline",
        "has_polygon",
        "rule_attribute_count",
    ]


  
def select_map_tokens(
    map_tokens: Sequence[MapToken],
    ego: EgoState,
    max_map_tokens: int,
) -> list[MapToken]:
    """Select nearby and rule-relevant map tokens deterministically."""
    return sorted(
        map_tokens,
        key=lambda token: _map_selection_key(token, ego),
    )[:max_map_tokens]


def _encode_map_token(token: MapToken, ego: EgoState) -> list[float]:
    points = _geometry_points(token)
    center_x, center_y = _centroid(points)
    rel_x = center_x - ego.x
    rel_y = center_y - ego.y
    distance = math.hypot(rel_x, rel_y)
    span = _geometry_span(points)
    traffic_state = token.traffic_state if token.traffic_state in TRAFFIC_STATE_CODES else "unknown"
    return [
        rel_x,
        rel_y,
        distance,
        float(len(points)),
        span,
        MAP_TYPE_CODES.get(token.type, 0.0),
        TRAFFIC_STATE_CODES[traffic_state],
        1.0 if token.polyline else 0.0,
        1.0 if token.polygon else 0.0,
        float(len(token.rule_attributes)),
    ]


def _map_selection_key(token: MapToken, ego: EgoState) -> tuple[float, float, str]:
    points = _geometry_points(token)
    center_x, center_y = _centroid(points)
    distance = math.hypot(center_x - ego.x, center_y - ego.y)
    risk_prior = MAP_RISK_PRIOR.get(token.type, 0.0)
    effective_distance = distance / (1.0 + risk_prior)
    return (effective_distance, distance, token.id)


def _geometry_points(token: MapToken) -> list[list[float]]:
    points: list[list[float]] = []
    if token.polyline:
        points.extend(token.polyline)
    if token.polygon:
        points.extend(token.polygon)
    return points


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    x_sum = sum(_number(point[0]) for point in points)
    y_sum = sum(_number(point[1]) for point in points)
    return x_sum / len(points), y_sum / len(points)


def _geometry_span(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [_number(point[0]) for point in points]
    ys = [_number(point[1]) for point in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _number(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("geometry point values must be numeric")
    return float(value)
