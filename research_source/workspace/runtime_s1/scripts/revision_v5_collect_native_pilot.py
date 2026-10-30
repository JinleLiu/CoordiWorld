"""Bounded, TRAIN-only real-data integration pilot; no optimizer or qualification.

Current inputs/controller and privileged evaluation are separate functions.
Output is a pilot evidence bundle, deliberately NOT a RevisionBatch native
export. No historical seed, label, exposure completion or anchor is invented.
"""
import argparse
import copy
from dataclasses import fields
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
import yaml

from coordiworld.revision_v5.contracts import canonical_fingerprint
from coordiworld.revision_v5.data import tensor_fingerprint
from coordiworld.revision_v5.native_adapter import (sha256_file, predict_current_controller_execution,
                                                   recompute_current_execution_rules)


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_train_scenes(roster, count):
    """First scene per drive in fixed input-only order; no response selection."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 8:
        raise ValueError("Pilot scene count must be 1..8")
    rows = roster.get("scenes", [])
    seen, roles = set(), {}
    for row in rows:
        if row["token"] in seen:
            raise ValueError("Duplicate token in frozen roster")
        seen.add(row["token"])
        if roles.setdefault(row["drive"], row["role"]) != row["role"]:
            raise ValueError("Frozen drive is assigned to multiple roles")
    result, drives = [], set()
    for row in sorted((r for r in rows if r["role"] == "train"), key=lambda r: (r["log"], r["index"])):
        if row["drive"] not in drives:
            result.append(row); drives.add(row["drive"])
        if len(result) == count:
            break
    if len(result) != count:
        raise ValueError("Insufficient distinct TRAIN drives in frozen roster")
    return result


def reset_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # UUID observation tokens are expressly excluded: the upstream IDM uses
    # uuid4 only for per-observation metadata. Stable track_token is the identity.
    value = {"seed": seed, "python_state_sha256": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
             "numpy_state_sha256": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
             "torch_cpu_state_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
             "uuid_observation_tokens_controlled": False, "identity": "track_token"}
    value["fingerprint"] = canonical_fingerprint(value)
    return value


def normalize_tracks(frames):
    result = []
    for frame in frames:
        rows = []
        for obj in frame.tracked_objects:
            velocity = getattr(obj, "velocity", None)
            rows.append({"track_token": str(obj.track_token), "type": obj.tracked_object_type.name,
                "box": [float(obj.center.x), float(obj.center.y), float(obj.center.heading),
                        float(obj.box.length), float(obj.box.width)],
                "velocity": [float(velocity.x), float(velocity.y)] if velocity is not None else None})
        if len({row["track_token"] for row in rows}) != len(rows):
            raise ValueError("Duplicate track identity in native teacher state")
        result.append(sorted(rows, key=lambda row: (row["track_token"], row["type"])))
    return result


def initialization_record(cache, source_hashes, traffic_name):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
    # This is the candidate-independent dynamic environment plus the privileged
    # exogenous future stream; no future field is exported as a decision input.
    lights = getattr(cache, "traffic_light_status", None)
    if lights is not None:
        raise ValueError("This pinned pilot expected absent cache traffic_light_status; add a lossless serializer before using a changed runtime")
    return {"definition": "candidate-independent initial dynamic state and fixed exogenous replay stream",
        "input_scope": "label-side only; includes privileged future replay; never passed to policy",
        "ego_state": ego_state_to_state_array(cache.ego_state).tolist(),
        "current_objects": normalize_tracks(cache.current_tracked_objects),
        "future_exogenous_objects": normalize_tracks(cache.future_tracked_objects),
        "traffic_light_status_attribute": None,
        "traffic_policy": traffic_name, "source_sha256": source_hashes,
        "route_lane_ids": list(cache.route_lane_ids),
        "map": {"name": cache.map_parameters.map_name, "version": cache.map_parameters.map_version},
        "uuid_policy": "observation UUID omitted; stable track_token, boxes and velocities retained"}


def response_diagnostics(states, valid, candidate_valid, first):
    states, valid, cv = np.asarray(states), np.asarray(valid), np.asarray(candidate_valid)
    if states.ndim != 4 or states.shape[-1] != 7 or valid.shape != states.shape[:-1] or cv.shape != states.shape[:1]:
        raise ValueError("Response tensors have incompatible shapes")
    if not cv[first] or valid.dtype != np.bool_ or cv.dtype != np.bool_:
        raise ValueError("Response masks/reference are invalid")
    common = valid & valid[first][None] & cv[:, None, None]
    common[first] = False
    distance = np.linalg.norm(states[..., :2] - states[first, ..., :2][None], axis=-1)[common]
    return {"reference_candidate": int(first), "paired_vehicle_time_cells": int(distance.size),
        "mean_position_response_m": float(distance.mean()) if distance.size else None,
        "max_position_response_m": float(distance.max()) if distance.size else None,
        "above_1mm_cells": int((distance > .001).sum()), "above_1cm_cells": int((distance > .01).sum()),
        "above_10cm_cells": int((distance > .1).sum()),
        "reference_response_max_m": 0.,
        "reference_is_zero_control_action": False,
        "diagnostic_thresholds_are_not_training_success_gates": True,
        "response_supported": bool(distance.size), "response_above_10cm_present": bool((distance > .1).any())}


def verify_runtime(registration):
    import navsim
    import nuplan
    result = {}
    for relative, expected in registration["dependency_commits"].items():
        root = Path(relative).resolve()
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError("Pinned dependency commit differs: " + relative)
        result[relative] = {"commit": actual, "root": str(root)}
    for module, relative in ((navsim, "third_party/navsim"), (nuplan, "third_party/nuplan-devkit")):
        origin = Path(inspect.getfile(module)).resolve()
        if not origin.is_relative_to(Path(relative).resolve()):
            raise ValueError("Imported dependency is outside the pinned checkout: " + str(origin))
    return result


def map_receipt(map_root):
    root = Path(map_root)
    manifest = root / "PUBLIC_DATA_MANIFEST.json"
    if not manifest.exists():
        manifest = root / "EXTRACTED.json"
    content = json.loads(manifest.read_text())
    if manifest.name == "PUBLIC_DATA_MANIFEST.json":
        asset = content.get("asset_spec", {})
        if (content.get("schema") != "coordiworld.public_data_extraction.v1" or content.get("asset") != "maps" or
                asset.get("layout") != "maps" or asset.get("path") != "public/nuplan-v1.1/nuplan-maps-v1.1.zip"):
            raise ValueError("Unsupported public map extraction manifest")
    elif content.get("map_api_version") != "nuplan-maps-v1.0":
        raise ValueError("Unsupported legacy map extraction manifest")
    files = content.get("files")
    if not files:
        raise ValueError("Missing verified map files")
    paths = [row["path"] for row in files]
    if len(paths) != len(set(paths)) or "nuplan-maps-v1.0.json" not in paths or not any(p.endswith("/map.gpkg") for p in paths):
        raise ValueError("Map v1.0 index or city geometry absent/duplicated")
    for row in content["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Map manifest path escapes map root")
        path = (Path(map_root) / relative).resolve()
        if not path.is_relative_to(Path(map_root).resolve()) or sha256_file(path) != row["sha256"]:
            raise ValueError("Map file hash mismatch")
        size = row.get("bytes", row.get("size"))
        if size is not None and path.stat().st_size != size:
            raise ValueError("Map file byte count mismatch")
    return {"manifest_sha256": sha256_file(manifest), "manifest_name": manifest.name,
            "verified_files": len(content["files"]), "map_api_version": "nuplan-maps-v1.0"}


def collect_scene(row, config, args, root, common_sources):
    from coordiworld.data.openscene import load_log, make_sample, save_sample
    from coordiworld.data.maps import CurrentMapProvider
    from coordiworld.data.pdm_proposer import PDMNominalProposer
    from coordiworld.navsim_adapter import OfficialComponentScorer
    from reactive_teacher_probe_v2 import extract_vehicle_tracks, teacher_origin, CapturingTraffic
    from diagnose_idm_path import retry_dropped_endpoint
    import nuplan.planning.simulation.observation.idm.idm_agent as idm

    current = root / row["token"]; current.mkdir()
    write_json(current / "requested_scene.json", row)
    log = Path(args.log_root) / (row["log"] + ".pkl")
    if not log.resolve().is_relative_to(Path(args.log_root).resolve()) or sha256_file(log) != row["sha256"]:
        raise ValueError("Raw log path/hash differs from frozen roster")
    frames = load_log(log); index = row["index"]
    if index < 3 or index + 8 >= len(frames):
        raise ValueError("Frozen scene lacks required history/future window")
    now = frames[index]
    if (str(now["token"]) != row["token"] or str(now["log_name"]) != row["log"] or
            int(now["timestamp"]) != row["timestamp"] or
            int(frames[index - 3]["timestamp"]) != row["window_start"] or
            int(frames[index + 8]["timestamp"]) != row["window_end"]):
        raise ValueError("Raw scene identity/window differs from frozen roster")
    history = frames[index - 3:index + 1]
    reset_rng(args.seed)
    tokens = CurrentMapProvider(args.map_root)(history, config["data"]["max_map_tokens"])
    bundle = PDMNominalProposer(args.map_root, config["candidates"])(history)
    scene, logged, metadata = make_sample(frames, index, config, tokens, bundle)
    if scene.candidates.shape[1:] != (12, 8, 3) or metadata["input_latest_timestamp"] != row["timestamp"]:
        raise ValueError("Frozen 12-slot candidate/history contract differs")
    metadata.update(data_role="train", drive_group=row["drive"], frozen_window_us=[row["window_start"], row["window_end"]])
    save_sample(current / "legacy_scene.npz", scene, logged, metadata)
    # This current-only computation happens BEFORE the privileged cache exists.
    execution, execution_01s, execution_receipt = predict_current_controller_execution(
        history, scene.candidates[0].numpy(), scene.candidate_valid[0].numpy())
    margins, rules = recompute_current_execution_rules(history, scene, metadata, execution, args.map_root)
    np.savez_compressed(current / "decision_inputs.npz",
        **{"input_" + f.name: getattr(scene, f.name).numpy() for f in fields(scene)},
        executed_candidates=execution[None], executed_rule_margins=margins[None], executed_rule_valid=rules[None])
    write_json(current / "input_metadata.json", metadata)

    traffic_name = "privileged_NAVSIM_IDM_endpoint_compat_v1"
    scorer = OfficialComponentScorer(args.map_root, current / "metric_cache", "reactive")
    capture = CapturingTraffic(scorer.traffic); scorer.traffic = capture
    cache = scorer.build_metric_cache(frames, index)
    initial = np.array([cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading])
    origin, frame_metadata = teacher_origin(now, initial)
    hashes = dict(common_sources, raw_log=row["sha256"], legacy_scene=sha256_file(current / "legacy_scene.npz"),
                  decision_inputs=sha256_file(current / "decision_inputs.npz"))
    init = initialization_record(cache, hashes, traffic_name)
    init_hash = canonical_fingerprint(init)
    write_json(current / "initialization.json", init)
    cv = scene.candidate_valid[0].numpy(); candidates = scene.candidates[0].numpy()
    eligible = ((scene.agent_class[0] == 1) & scene.agent_valid[0, -1]).numpy()
    m, n = len(candidates), scene.agents.shape[2]
    states = np.zeros((m, 41, n, 7), dtype=np.float32); valid = np.zeros((m, 41, n), dtype=bool)
    labels = {name: np.full(m, np.nan, dtype=np.float32) for name in ("at_fault_collision", "utility", "progress")}
    scores, pair_receipts, case_rows = {}, [], []
    first = int(np.flatnonzero(cv)[0]); events = []
    original = idm.create_path_from_se2
    idm.create_path_from_se2 = lambda poses: retry_dropped_endpoint(original, poses, events, True)
    try:
        for candidate in np.flatnonzero(cv):
            trial_cache = copy.deepcopy(cache)
            before = canonical_fingerprint(initialization_record(trial_cache, hashes, traffic_name))
            if before != init_hash:
                raise ValueError("Simulator initial state/exogenous stream mutated across candidates")
            rng = reset_rng(args.seed); capture.calls.clear(); events.clear(); started = time.monotonic()
            score = scorer.score(trial_cache, candidates[candidate])
            if not 1 <= len(capture.calls) <= 2:
                raise ValueError("Unexpected candidate/human traffic call count")
            ego, detections = capture.calls[0]
            if ego.shape != (41, 11) or len(detections) != 41:
                raise ValueError("Unexpected native simulation horizon")
            error = float(np.max(np.abs(ego - execution_01s[int(candidate)])))
            if error > 1e-7:
                raise ValueError("Current-only controller and teacher execution differ: " + str(error))
            target, mask = extract_vehicle_tracks(detections, metadata["tracks"], eligible, origin)
            if not np.array_equal(mask[0], eligible):
                raise ValueError("Teacher initial retained-vehicle coverage differs")
            current_error = float(np.max(np.abs(target[0, eligible] - scene.agents[0, -1].numpy()[eligible]))) if eligible.any() else 0.
            if current_error > 1e-3:
                raise ValueError("Teacher and current input actor states differ")
            states[candidate], valid[candidate] = target, mask
            scores[str(candidate)] = score
            labels["at_fault_collision"][candidate] = float(score["no_at_fault_collisions"] < 1.)
            labels["utility"][candidate] = score["single_stage_composite_excluding_two_frame_comfort"]
            labels["progress"][candidate] = score["ego_progress"]
            receipt = {"candidate": int(candidate), "reference_candidate": first,
                "initialization_sha256": init_hash, "rng": rng, "traffic_policy": traffic_name,
                "action_sha256": tensor_fingerprint(torch.from_numpy(execution[candidate])),
                "reference_action_sha256": tensor_fingerprint(torch.from_numpy(execution[first])),
                "current_state_max_error": current_error, "controller_state_max_error": error,
                "captured_environment_calls": len(capture.calls), "endpoint_compat_events": copy.deepcopy(events),
                "seconds": time.monotonic() - started}
            pair_receipts.append(receipt); case_rows.append(receipt)
            write_json(current / f"candidate-{candidate:02d}.json", {"receipt": receipt, "scores": score})
            gc.enable(); gc.collect()
        if canonical_fingerprint(initialization_record(cache, hashes, traffic_name)) != init_hash:
            raise ValueError("Simulator initial state/exogenous stream changed after candidates")
        repeated_rng = reset_rng(args.seed); capture.calls.clear(); events.clear()
        repeat_score = scorer.score(copy.deepcopy(cache), candidates[first])
        repeat, repeat_valid = extract_vehicle_tracks(capture.calls[0][1], metadata["tracks"], eligible, origin)
        repeat_delta = np.linalg.norm(repeat[..., :2] - states[first, ..., :2], axis=-1)
        repeat_error = float(repeat_delta[repeat_valid & valid[first]].max()) if (repeat_valid & valid[first]).any() else None
        repeated = bool(np.array_equal(repeat, states[first]) and np.array_equal(repeat_valid, valid[first]) and repeat_score == scores[str(first)])
        if not repeated or repeated_rng != pair_receipts[0]["rng"]:
            raise ValueError("Same-initialization same-seed reference repeat is not exact")
    finally:
        idm.create_path_from_se2 = original; gc.enable()
    sample = np.arange(5, 41, 5)
    truth, truth_valid = states[:, sample], valid[:, sample]
    common = truth_valid & truth_valid[first][None] & cv[:, None, None]
    delta = truth - truth[first][None]
    delta[..., 2] = np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2]))
    delta[~common] = 0.
    np.savez_compressed(current / "teacher.npz", states_01s=states, valid_01s=valid,
        states_05s=truth, valid_05s=truth_valid, paired_response_delta=delta, paired_response_valid=common,
        candidate_valid=cv, eligible_current_vehicle=eligible, sample_indices=sample,
        **{"simulated_ego_" + str(j): value for j, value in execution_01s.items()},
        **labels, label_valid=cv)
    report = {"scene_id": row["token"], "log_name": row["log"], "drive_id": row["drive"], "split": "train",
        "scope": "real TRAIN native integration pilot; privileged simulator response; no model training or qualification",
        "candidate_sha256": metadata["candidate_manifest"]["sha256"], "slots": m,
        "valid_candidates": int(cv.sum()), "current_retained_vehicles": int(eligible.sum()),
        "candidate_fallback": metadata["candidate_manifest"].get("fallback"), "map_tokens": len(tokens),
        "tracks": metadata["tracks"], "coordinate_frame": frame_metadata,
        "initialization_sha256": init_hash, "pair_receipts": pair_receipts,
        "execution_receipt": dict(execution_receipt, native_teacher_parity_verified=True),
        "reference_repeated_exact": repeated, "reference_repeat_max_position_error_m": repeat_error,
        "response_diagnostics": response_diagnostics(truth, truth_valid, cv, first),
        "physical_collision_label": None, "first_event_labels": None, "existence_labels": None,
        "utility_namespace": "single_stage_composite_excluding_two_frame_comfort",
        "human_filter": "official_after_filter", "traffic_policy": traffic_name,
        "native_export_ready": False, "qualified_anchor": None, "complete_B_F_exposure_audit": None,
        "optimizer_updates": 0, "input_files_sha256": hashes,
        "teacher_npz_sha256": sha256_file(current / "teacher.npz")}
    write_json(current / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="configs/group_cache_execution_v2.json")
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-scene-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.seed < 0 or not 60 <= args.max_seconds <= 7200 or not 30 <= args.max_scene_seconds <= args.max_seconds:
        parser.error("Use nonnegative seed, 60..7200 total seconds and 30..total scene seconds")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    requested, successes, failures, skipped = [], [], [], []
    registration = json.loads(Path(args.registration).read_text())
    if sha256_file(registration["roster"]) != registration["roster_sha256"] or sha256_file(registration["config"]) != registration["config_sha256"]:
        raise ValueError("Frozen roster/config source differs from registration")
    config = yaml.safe_load(Path(registration["config"]).read_text())
    if config["candidates"]["nominal"] != "pdm_closed_current_v1" or config["data"]["future_steps"] != 8 or config["data"]["history_frames"] != 4:
        raise ValueError("Use the fixed current PDM 12-slot data contract")
    requested = select_train_scenes(json.loads(Path(registration["roster"]).read_text()), args.scenes)
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(args.map_root).resolve())
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    torch.use_deterministic_algorithms(True)
    dependencies = verify_runtime(registration)
    maps = map_receipt(args.map_root)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Pilot needs at least 2 GiB free output space")
    sources = {str(p): sha256_file(p) for p in sorted(Path("src/coordiworld").rglob("*.py"))}
    for name in ("scripts/revision_v5_collect_native_pilot.py", "scripts/reactive_teacher_probe_v2.py", "scripts/diagnose_idm_path.py"):
        sources[name] = sha256_file(name)
    common = {"roster": registration["roster_sha256"], "config": registration["config_sha256"],
              "map_manifest": maps["manifest_sha256"], "source_manifest": canonical_fingerprint(sources)}
    write_json(root / "plan.json", {"format": "coordiworld_native_pilot_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested": requested, "selection": "first scene per distinct TRAIN drive ordered by log,index; no outcome replacement",
        "seed": args.seed, "max_seconds": args.max_seconds, "max_scene_seconds": args.max_scene_seconds,
        "precision": "float32 tensors; native simulator float64", "workers": 1, "optimizer_updates": 0,
        "dependencies": dependencies, "maps": maps, "sources": sources,
        "formal_execution_ready": False, "actual_model_performance": None})
    previous_handler = signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("Frozen pilot wall-clock budget reached")))
    try:
        for row in requested:
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 1:
                skipped.append({"scene_id": row["token"], "reason": "total wall-clock budget"})
                continue
            signal.setitimer(signal.ITIMER_REAL, min(args.max_scene_seconds, remaining))
            try:
                result = collect_scene(row, config, args, root, common)
                successes.append(result)
                print(json.dumps({"scene": row["token"], "status": "collected", "diagnostics": result["response_diagnostics"]}), flush=True)
            except Exception as error:
                failure = {"scene_id": row["token"], "drive_id": row["drive"], "error": repr(error), "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(root / (row["token"] + ".FAILURE.json"), failure)
                print(json.dumps({"scene": row["token"], "status": "failed", "error": repr(error)}), flush=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0); gc.enable(); gc.collect()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)
        changed_sources = [str(p) for p, digest in sources.items() if sha256_file(p) != digest]
        final = {"format": "coordiworld_native_pilot_v1", "requested_scenes": len(requested),
            "completed_scenes": len(successes), "failed_scenes": len(failures), "budget_skipped_scenes": len(skipped),
            "success_scene_ids": [r["scene_id"] for r in successes], "failures": failures, "skipped": skipped,
            "source_hashes_unchanged": not changed_sources, "changed_sources": changed_sources,
            "elapsed_seconds": time.monotonic() - started, "optimizer_updates": 0,
            "native_export_ready": False, "actual_model_performance": None,
            "integration_checks_passed_for_all_requested": len(successes) == len(requested) and not changed_sources,
            "newly_collected_observations_are_not_expected_values": True,
            "no_outcome_replacements": True}
        write_json(root / "report.json", final)
        manifest_rows = []
        lookup = {r["token"]: r for r in requested}
        for receipt in successes:
            sid = receipt["scene_id"]; row = lookup[sid]
            relative = sid + "/legacy_scene.npz"
            manifest_rows.append({"scene_id": sid, "log_name": row["log"], "drive": row["drive"],
                "split": "train", "path": relative, "cache_sha256": sha256_file(root / relative),
                "candidate_sha256": receipt["candidate_sha256"], "valid_candidates": receipt["valid_candidates"]})
        source_logs = {r["log"]: r["sha256"] for r in requested}
        write_json(root / "legacy_manifest.json", {"protocol": "revision_v5_native_pilot_cache_v1",
            "source": "openscene", "label_provider": "logged_box_replay_plus_heading_progress_v1",
            "official_epdms": False, "config": config, "scenes": manifest_rows,
            "source_files": [{"name": name + ".pkl", "sha256": digest} for name, digest in sorted(source_logs.items())],
            "requested_scenes": requested, "failures": failures, "skipped": skipped,
            "intended_use": "native integration and low-sample engineering check only; no anchor qualification"})
    if not final["integration_checks_passed_for_all_requested"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
