"""Fail-closed data/provenance contracts for the prospective revision.

These records describe evidence; they never promote a development exposure to
an independent confirmation sample. Old result files are deliberately untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Iterable, Mapping


SPLITS = frozenset({"train", "development", "cal_candidate", "cal_selected", "confirmation"})
COLLISION_EVENTS = frozenset({"physical_contact", "at_fault_collision"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_fingerprint(value) -> str:
    """Hash JSON-compatible configuration, forbidding nonfinite JSON values."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class LabelSemantics:
    input_provider: str
    label_provider: str
    candidate_sha256: str
    execution_policy: str
    traffic_policy: str
    human_filter: str
    collision_event: str
    violation_event: str
    evaluator_version: str
    coordinate_frame: str = "current_ego_rear_axle"
    time_convention: str = "seconds_from_current_observation"

    def validate(self) -> "LabelSemantics":
        for name, value in asdict(self).items():
            _nonempty(value, name)
        _sha(self.candidate_sha256, "candidate_sha256")
        if self.collision_event not in COLLISION_EVENTS:
            raise ValueError("collision_event must distinguish physical_contact and at_fault_collision")
        if self.human_filter not in {"none", "official_before_filter", "official_after_filter"}:
            raise ValueError("human_filter must explicitly identify the official filtering stage")
        if self.collision_event == "physical_contact" and self.human_filter == "official_after_filter":
            raise ValueError("Raw physical contact cannot be redefined by official post-filter exemptions")
        return self

    @property
    def fingerprint(self) -> str:
        self.validate()
        return canonical_fingerprint(asdict(self))


def require_label_compatibility(actual: LabelSemantics, expected: LabelSemantics) -> None:
    """Use before pooling labels or comparing methods under a shared protocol."""
    actual.validate()
    expected.validate()
    differences = [k for k, v in asdict(actual).items() if v != asdict(expected)[k]]
    if differences:
        raise ValueError("Incompatible label/input semantics: " + ", ".join(differences))


@dataclass(frozen=True)
class ExposureRecord:
    scene_id: str
    drive_id: str
    split: str
    source_line: str

    def validate(self) -> "ExposureRecord":
        for name, value in asdict(self).items():
            _nonempty(value, name)
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split {self.split!r}")
        return self


@dataclass(frozen=True)
class DatasetManifest:
    experiment_id: str
    semantics: LabelSemantics
    records: tuple[ExposureRecord, ...]
    # An explicit availability declaration, including derived features. Names
    # alone cannot establish whether a feature is safe.
    input_availability: Mapping[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        validate_manifest(self)
        return canonical_fingerprint(asdict(self))


def validate_manifest(manifest: DatasetManifest, exposure_history: Iterable[ExposureRecord] = ()) -> dict:
    _nonempty(manifest.experiment_id, "experiment_id")
    manifest.semantics.validate()
    if not manifest.records or not manifest.input_availability:
        raise ValueError("Manifest requires scene records and explicit input availability")
    allowed_inputs = {"observed_current_or_history", "static_map", "candidate_plan", "candidate_controller_rollout"}
    for name, availability in manifest.input_availability.items():
        _nonempty(name, "input name")
        if availability not in allowed_inputs:
            raise ValueError(f"Input {name!r} is not available at decision time: {availability!r}")
    seen_scenes: set[str] = set()
    for record in manifest.records:
        record.validate()
        if record.scene_id in seen_scenes:
            raise ValueError(f"Duplicate scene in manifest: {record.scene_id}")
        seen_scenes.add(record.scene_id)
    drive_roles: dict[str, str] = {}
    scene_identity: dict[str, tuple[str, str]] = {}
    for record in (*tuple(exposure_history), *manifest.records):
        record.validate()
        prior_role = drive_roles.setdefault(record.drive_id, record.split)
        if prior_role != record.split:
            raise ValueError(f"Drive exposure leaks across splits/lines: {record.drive_id}: {prior_role} vs {record.split}")
        identity = (record.drive_id, record.split)
        prior_identity = scene_identity.setdefault(record.scene_id, identity)
        if prior_identity != identity:
            raise ValueError(f"Scene identity/split changed across lines: {record.scene_id}")
    return {
        "scene_count": len(manifest.records),
        "drive_count": len({r.drive_id for r in manifest.records}),
        "split_scene_counts": {s: sum(r.split == s for r in manifest.records) for s in sorted(SPLITS)},
        "semantics_fingerprint": manifest.semantics.fingerprint,
    }


@dataclass(frozen=True)
class PairedResponseRecord:
    pair_id: str
    scene_id: str
    simulator_domain: str
    reference_simulator_domain: str
    initial_state_sha256: str
    reference_initial_state_sha256: str
    traffic_policy: str
    reference_traffic_policy: str
    random_seed: int
    reference_random_seed: int
    action_sha256: str
    reference_action_sha256: str
    target_kind: str = "within_simulator_difference"
    future_available_to_model: bool = False


def validate_paired_responses(records: Iterable[PairedResponseRecord]) -> dict:
    records = tuple(records)
    if not records:
        raise ValueError("A paired response dataset must contain records")
    seen: set[str] = set()
    for record in records:
        for name, value in asdict(record).items():
            if name.endswith("sha256"):
                _sha(value, name)
        for name in ("pair_id", "scene_id", "simulator_domain", "reference_simulator_domain", "traffic_policy", "reference_traffic_policy"):
            _nonempty(getattr(record, name), name)
        if record.pair_id in seen:
            raise ValueError(f"Duplicate response pair: {record.pair_id}")
        seen.add(record.pair_id)
        for left, right in (("simulator_domain", "reference_simulator_domain"), ("initial_state_sha256", "reference_initial_state_sha256"), ("traffic_policy", "reference_traffic_policy"), ("random_seed", "reference_random_seed")):
            if getattr(record, left) != getattr(record, right):
                raise ValueError(f"Response target is not a controlled same-domain pair: {left}")
        for seed in (record.random_seed, record.reference_random_seed):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Response pair seed must be a nonnegative integer")
        if record.target_kind != "within_simulator_difference" or record.future_available_to_model:
            raise ValueError("Response targets must remain same-domain supervision, never decision inputs")
    return {"pairs": len(records), "domains": sorted({r.simulator_domain for r in records}), "claim_scope": "simulator_intervention_only"}
