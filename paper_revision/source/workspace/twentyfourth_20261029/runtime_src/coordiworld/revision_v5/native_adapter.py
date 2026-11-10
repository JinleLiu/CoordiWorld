"""Read existing OpenScene/IDM archives without promoting them to V5 evidence.

The legacy schemas are defined by data/openscene.py and
scripts/teacher_density_worker.py. Controller predictions are recomputed from
history only; legacy teacher arrays remain label-side diagnostics. A successful
inspection is not a native export, a complete exposure audit, or qualification.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..schema import SceneInputs


LEGACY_SAMPLE_INDICES = np.arange(5, 41, 5)
NATIVE_GAPS = (
    "New paired simulator collection must record initialization and RNG provenance; do not backfill historical seeds.",
    "Recompute controller execution using current/history inputs and compare it with captured teacher ego execution.",
    "Recompute executed rule margins from current map polygons; cached planned-path margins cannot be reused after execution changes.",
    "Resolve complete B/F drive exposure inventories before assigning V5 development/calibration/confirmation roles.",
    "Independently qualify an action-independent factual anchor; old action-conditioned full_world is not such an anchor.",
    "Retained-vehicle teacher tracks do not provide all-entity physical collision or first-event labels.",
    "Official legacy single-stage components are not complete EPDMS; preserve their evaluator/filter/traffic semantics.",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Archive members must stay inside their supplied root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Archive member escapes its supplied root")
    return path


def _identity(metadata, record):
    for key in ("scene_id", "log_name"):
        if not isinstance(record.get(key), str) or not record[key] or metadata.get(key) != record[key]:
            raise ValueError("Scene identity differs: " + key)
    digest = metadata.get("candidate_manifest", {}).get("sha256", metadata.get("candidate_sha256"))
    if digest != record.get("candidate_sha256") or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Candidate fingerprint differs")


def read_legacy_inputs(path, record):
    """Load only input_* and metadata_json; no target_* arrays are accessed."""
    with np.load(path, allow_pickle=False) as archive:
        names = {"input_" + field.name for field in fields(SceneInputs)}
        if {name for name in archive.files if name.startswith("input_")} != names:
            raise ValueError("Legacy input fields differ from SceneInputs")
        scene = SceneInputs(**{field.name: torch.from_numpy(archive["input_" + field.name].copy())
                               for field in fields(SceneInputs)})
        metadata = json.loads(str(archive["metadata_json"]))
    scene.validate()
    if scene.ego.shape[0] != 1:
        raise ValueError("Legacy cache must contain exactly one scene")
    _identity(metadata, record)
    tracks = metadata.get("tracks")
    n = scene.agents.shape[2]
    if (not isinstance(tracks, list) or len(tracks) > n or
            any(not isinstance(t, str) or not t for t in tracks) or len(set(tracks)) != len(tracks)):
        raise ValueError("Legacy actor slots require unique nonempty track IDs")
    if scene.agent_valid[0, :, len(tracks):].any():
        raise ValueError("Observed actor lacks an identity")
    if metadata.get("input_latest_timestamp") != metadata.get("timestamp") or metadata.get("timestamp") is None:
        raise ValueError("Legacy current input timestamp does not match scene timestamp")
    return scene, metadata


def global_execution_to_local(states, origin, sample_indices=LEGACY_SAMPLE_INDICES):
    """Pinned NAVSIM StateIndex 0:3 is global rear-axle x/y/heading."""
    states, origin, indices = np.asarray(states), np.asarray(origin), np.asarray(sample_indices)
    if (states.shape != (41, 11) or origin.shape != (3,) or indices.shape != (8,) or
            not np.array_equal(indices, LEGACY_SAMPLE_INDICES) or
            not np.isfinite(states).all() or not np.isfinite(origin).all()):
        raise ValueError("Expected pinned 41 x 11 controller states and 0.5..4s samples")
    delta = states[indices, :2] - origin[:2]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    local_xy = delta @ np.array([[c, -s], [s, c]])
    local_yaw = np.arctan2(np.sin(states[indices, 2] - origin[2]), np.cos(states[indices, 2] - origin[2]))
    return np.c_[local_xy, local_yaw].astype(np.float32)


def predict_current_controller_execution(history, candidates, candidate_valid):
    """Run the pinned controller without metric caches or future observations.

    Requires the actual pinned NAVSIM/nuPlan runtime. This function converts
    only the current ego status, and never builds a metric cache or calls a
    traffic policy. Server parity against
    the teacher's captured first ego rollout remains a required integration test.
    """
    candidates, candidate_valid = np.asarray(candidates), np.asarray(candidate_valid)
    if (not isinstance(history, (list, tuple)) or len(history) != 4 or
            candidates.ndim != 3 or candidates.shape[1:] != (8, 3) or
            candidate_valid.shape != candidates.shape[:1] or candidate_valid.dtype != np.bool_ or
            not candidate_valid.any() or not np.isfinite(candidates).all()):
        raise ValueError("Controller expects four history frames and finite M x 8 x 3 candidates")
    times = np.asarray([frame["timestamp"] for frame in history], dtype=np.float64)
    if not np.allclose(np.diff(times) / 1e6, .5, atol=.08):
        raise ValueError("History timestamps must be ordered and contiguous")
    from navsim.common.dataclasses import Scene, Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    # Use the same pinned conversion used by Scene.from_scene_dict_list, without
    # its unrelated map/sensor/annotation construction and without any future.
    ego_status = Scene._build_ego_status(history[-1])
    initial = ego_status_to_ego_state(ego_status, get_pacifica_parameters(), TimePoint(int(times[-1])))
    simulation_sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    prediction_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    simulator = PDMSimulator(simulation_sampling)
    origin = np.array([initial.rear_axle.x, initial.rear_axle.y, initial.rear_axle.heading])
    local = np.zeros_like(candidates, dtype=np.float32)
    full_states = {}
    for index in np.flatnonzero(candidate_valid):
        trajectory = transform_trajectory(Trajectory(candidates[index].astype(np.float64), prediction_sampling), initial)
        proposal = get_trajectory_as_array(trajectory, simulation_sampling, initial.time_point)
        executed = simulator.simulate_proposals(proposal[None], initial)[0]
        local[index] = global_execution_to_local(executed, origin)
        full_states[int(index)] = executed
    return local, full_states, {
        "input_latest_timestamp": int(times[-1]), "num_future_frames": 0,
        "uses_metric_cache": False, "uses_traffic_teacher": False,
        "coordinate_origin": origin.tolist(), "native_teacher_parity_verified": False,
    }


def inspect_legacy_teacher(scene, metadata, record, teacher_npz, teacher_json):
    teacher_meta = json.loads(Path(teacher_json).read_text())
    _identity(teacher_meta, record)
    if teacher_meta.get("tracks") != metadata["tracks"]:
        raise ValueError("Teacher actor identity ordering differs from input slots")
    if teacher_meta.get("split") != record["split"] or teacher_meta.get("repeated_first_exact") is not True:
        raise ValueError("Teacher split or historical repeat check differs")
    m, h = scene.candidates.shape[1:3]
    n = scene.agents.shape[2]
    if h != 8:
        raise ValueError("Legacy teacher uses exactly eight 0.5-second future samples")
    cv = scene.candidate_valid[0].numpy()
    expected_eligible = (scene.agent_class[0].numpy() == 1) & scene.agent_valid[0, -1].numpy()
    if teacher_meta.get("first_valid") != int(np.flatnonzero(cv)[0]):
        raise ValueError("Teacher reference differs from first valid candidate")
    if not np.array_equal(np.asarray(teacher_meta.get("sample_times_s")), np.arange(1, 9) * .5):
        raise ValueError("Teacher sample times differ")
    origin = np.asarray(teacher_meta.get("coordinate_frame", {}).get("simulator_origin"), dtype=float)
    with np.load(teacher_npz, allow_pickle=False) as archive:
        states = archive["states_01s"].copy()
        valid = archive["valid_01s"].copy()
        if (states.shape != (m, 41, n, 7) or valid.shape != (m, 41, n) or valid.dtype != np.bool_ or
                not np.isfinite(states).all() or not np.array_equal(archive["candidate_valid"], cv) or
                not np.array_equal(archive["eligible_current_vehicle"], expected_eligible) or
                not np.array_equal(archive["sample_indices"], LEGACY_SAMPLE_INDICES)):
            raise ValueError("Teacher tensor shape, mask, or sampling differs")
        if valid[~cv].any() or valid[..., ~expected_eligible].any():
            raise ValueError("Unavailable candidate or ineligible actor has teacher labels")
        if not np.array_equal(valid[cv, 0], np.broadcast_to(expected_eligible, (int(cv.sum()), n))):
            raise ValueError("Teacher initial coverage differs from current vehicle inputs")
        if not np.array_equal(archive["states_05s"], states[:, LEGACY_SAMPLE_INDICES]) or not np.array_equal(archive["valid_05s"], valid[:, LEGACY_SAMPLE_INDICES]):
            raise ValueError("Teacher half-second targets differ from stored fine sampling")
        current_error = (float(np.max(np.abs(states[cv, 0][:, expected_eligible] - scene.agents[0, -1].numpy()[expected_eligible])))
                         if expected_eligible.any() else 0.)
        if current_error > 1e-3:
            raise ValueError("Teacher current states differ from observed actor states")
        execution_distance = []
        for index in np.flatnonzero(cv):
            simulated = archive["simulated_ego_" + str(index)]
            executed = global_execution_to_local(simulated, origin)
            if not np.allclose(simulated[0, :3], origin, atol=1e-5, rtol=0):
                raise ValueError("Captured execution initial pose differs from teacher origin")
            execution_distance.extend(np.linalg.norm(executed[:, :2] - scene.candidates[0, index, :, :2].numpy(), axis=-1).tolist())
    first = int(np.flatnonzero(cv)[0])
    truth, mask = states[:, LEGACY_SAMPLE_INDICES], valid[:, LEGACY_SAMPLE_INDICES]
    common = mask & mask[first][None] & cv[:, None, None]
    common[first] = False
    distance = np.linalg.norm(truth[..., :2] - truth[first, ..., :2][None], axis=-1)[common]
    return {
        "retained_vehicle_count": int(expected_eligible.sum()), "candidate_count": int(cv.sum()),
        "reference_candidate": first, "actor_identity_checked": True,
        "current_state_max_error": current_error, "historical_repeat_flag": True,
        "paired_response_cells": int(distance.size),
        "teacher_response_mean_m": float(distance.mean()) if distance.size else None,
        "teacher_response_over_10cm_cells": int((distance > .1).sum()),
        "planned_vs_captured_execution_mean_m": float(np.mean(execution_distance)),
        "planned_vs_captured_execution_max_m": float(np.max(execution_distance)),
        "complete_initialization_or_rng_receipt": False,
        "decision_time_controller_parity": None,
        "source_scope": "historical privileged simulator; retained vehicles only; not real counterfactual truth",
    }


def inspect_current_controller_parity(scene, metadata, record, raw_log, expected_log_sha256, teacher_npz):
    """Compare newly computed decision-time execution with the saved teacher.

    A raw log contains future annotations, but only its selected four history
    frames are passed to the controller function. No observed future controls
    or teacher states are passed into that function.
    """
    from ..data.openscene import load_log
    digest = sha256_file(raw_log)
    if digest != expected_log_sha256:
        raise ValueError("Raw log SHA-256 differs from the legacy source manifest")
    frames = load_log(raw_log)
    indices = [index for index, frame in enumerate(frames) if str(frame["token"]) == record["scene_id"]]
    if len(indices) != 1 or indices[0] < 3:
        raise ValueError("Missing or duplicated current scene in the raw log")
    index = indices[0]
    history = frames[index - 3:index + 1]
    if str(history[-1]["log_name"]) != record["log_name"] or int(history[-1]["timestamp"]) != metadata["timestamp"]:
        raise ValueError("Raw current frame identity differs from the cached input")
    local, executed, receipt = predict_current_controller_execution(
        history, scene.candidates[0].numpy(), scene.candidate_valid[0].numpy())
    errors = []
    with np.load(teacher_npz, allow_pickle=False) as archive:
        for candidate, value in executed.items():
            stored = archive["simulated_ego_" + str(candidate)]
            if stored.shape != value.shape:
                raise ValueError("New and captured controller state shapes differ")
            errors.append(float(np.max(np.abs(stored - value))))
    # This is a fixed numerical parity tolerance, not a score-dependent gate.
    passed = all(error <= 1e-7 for error in errors)
    receipt.update(native_teacher_parity_verified=passed,
                   compared_candidates=len(errors), state_max_absolute_errors=errors,
                   parity_absolute_tolerance=1e-7, raw_log_sha256=digest,
                   execution_input_origin="new current-only controller computation; teacher used only for comparison")
    return receipt


def recompute_current_execution_rules(history, scene, metadata, executed_candidates, map_root):
    """Use current map polygons after checking token identity against the cache.

    Future teacher arrays cannot be accepted by this API. Caller must pass the
    output of the current-only controller after native parity verification.
    """
    from ..data.maps import CurrentMapProvider
    from ..geometry import map_rule_margins
    candidates = np.asarray(executed_candidates)
    if candidates.shape != tuple(scene.candidates.shape[1:]) or not np.isfinite(candidates).all():
        raise ValueError("Executed candidates must match one-scene candidate shape")
    if len(history) != scene.ego.shape[1] or int(history[-1]["timestamp"]) != metadata["timestamp"]:
        raise ValueError("Rule geometry requires the exact current history")
    p = scene.map_features.shape[1]
    tokens = CurrentMapProvider(map_root)(history, p)
    identifiers = [token["id"] for token in tokens]
    prior = [token.get("id") for token in metadata.get("map_token_metadata", [])]
    if identifiers != prior:
        raise ValueError("Reconstructed current map tokens differ from the cached token identities")
    features = np.zeros((p, 8), dtype=np.float32)
    valid = np.zeros(p, dtype=bool)
    for i, token in enumerate(tokens):
        features[i] = token["features"]
        valid[i] = token.get("available", True)
    if not np.allclose(features, scene.map_features[0].numpy(), atol=1e-5, rtol=0) or not np.array_equal(valid, scene.map_valid[0].numpy()):
        raise ValueError("Reconstructed map geometry features differ from cached current inputs")
    m, h = candidates.shape[:2]
    margins = np.zeros((m, h, p, 4), dtype=np.float32)
    available = np.zeros_like(margins, dtype=bool)
    if tokens:
        values, masks = map_rule_margins(candidates, scene.ego[0, -1, 5:7].numpy(),
                                        float(scene.ego_rear_axle_to_center[0]), tokens)
        margins[:, :, :len(tokens)], available[:, :, :len(tokens)] = values, masks
    return margins, available


def inspect_legacy_bundle(manifest_path, *, teacher_root=None, scene_ids=None, max_scenes=4, log_root=None):
    """Read a bounded input-selected TRAIN subset; never open held-out tensors."""
    if isinstance(max_scenes, bool) or not isinstance(max_scenes, int) or not 1 <= max_scenes <= 1024:
        raise ValueError("Use an explicit bounded inspection of 1..1024 scenes")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("scenes")
    if not isinstance(records, list) or not records:
        raise ValueError("Missing legacy scene records")
    index = {row["scene_id"]: row for row in records}
    if len(index) != len(records):
        raise ValueError("Duplicate legacy scene IDs")
    if scene_ids is None:
        selected = [row for row in records if row["split"] == "train"][:max_scenes]
    else:
        scene_ids = list(scene_ids)
        if not scene_ids or len(scene_ids) > max_scenes or len(set(scene_ids)) != len(scene_ids) or any(sid not in index for sid in scene_ids):
            raise ValueError("Invalid explicit inspection scene IDs")
        selected = [index[sid] for sid in scene_ids]
    if not selected or any(row["split"] != "train" for row in selected):
        raise ValueError("Legacy inspection opens only original TRAIN tensors")
    if log_root is not None and teacher_root is None:
        raise ValueError("Controller parity requires an explicit teacher root")
    log_hashes = {row["name"]: row["sha256"] for row in manifest.get("source_files", [])}
    report = {
        "format": "coordiworld_revision_v5_legacy_inspection_v1",
        "manifest_sha256": sha256_file(manifest_path), "selection": "input-only existing record order or explicit IDs",
        "native_export_ready": False, "model_forward_executed": False, "optimizer_updates": 0,
        "actual_scientific_results": None, "inspection_passed": True, "scenes": [],
        "unresolved_native_requirements": list(NATIVE_GAPS),
    }
    for row in selected:
        item = {"scene_id": row["scene_id"], "log_name": row["log_name"], "original_split": row["split"]}
        try:
            path = _inside(manifest_path.parent, row["path"])
            digest = sha256_file(path)
            if row.get("cache_sha256") not in (None, digest):
                raise ValueError("Cache SHA-256 differs from manifest")
            scene, metadata = read_legacy_inputs(path, row)
            item.update(cache_sha256=digest, input_schema_valid=True,
                        independent_confidence_available=bool(scene.confidence_valid.any()),
                        input_actor_source_channels=torch.unique(scene.source[scene.agent_valid], dim=0).tolist())
            if teacher_root is not None:
                teacher_npz = _inside(teacher_root, row["scene_id"] + ".npz")
                teacher_json = _inside(teacher_root, row["scene_id"] + ".json")
                item.update(teacher_npz_sha256=sha256_file(teacher_npz), teacher_json_sha256=sha256_file(teacher_json),
                            teacher=inspect_legacy_teacher(scene, metadata, row, teacher_npz, teacher_json))
                if log_root is not None:
                    log_name = row["log_name"] + ".pkl"
                    raw_log = _inside(log_root, log_name)
                    item["current_controller_parity"] = inspect_current_controller_parity(
                        scene, metadata, row, raw_log, log_hashes[log_name], teacher_npz)
                    if not item["current_controller_parity"]["native_teacher_parity_verified"]:
                        raise ValueError("Current-only controller differs from captured teacher execution")
            item["inspection_passed"] = True
        except (ValueError, KeyError, OSError, AssertionError, IndexError, TypeError) as error:
            report["inspection_passed"] = False
            item.update(inspection_passed=False, error=f"{type(error).__name__}: {error}")
        report["scenes"].append(item)
    report["inspected_scenes"] = len(selected)
    report["inspected_logs"] = len({row["log_name"] for row in selected})
    return report
