"""Rear-axle-corrected IDM v2 paired teachers on full-ROI B TRAIN and B DEV.

No training, fabricated native-export qualification, physical contact label or
full EPDMS is produced. Natural and geometry-enriched samples remain separate.
The original four-scene native pilot script and artifacts are not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import fields
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch

import revision_v5_collect_native_pilot as pilot
from coordiworld.revision_v5.contracts import canonical_fingerprint
from coordiworld.revision_v5.data import tensor_fingerprint
from coordiworld.revision_v5.native_adapter import read_legacy_inputs, predict_current_controller_execution, sha256_file
from coordiworld.revision_v5.native_sampling import current_geometry_screen, select_cohorts
from coordiworld.revision_v5.idm_teacher_geometry import DOMAIN, audit_native_builder, install_corrected_builder


SCHEMA = "coordiworld.paired_teacher_collection.v1"
TEACHER_DOMAIN = DOMAIN
FROZEN_ROSTER_SHA = "9b930a7da8ac2bf8a4012544c66400b6d5788aa30bc5ac96772937484fd6ce6d"


def inside(root, member):
    relative = Path(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Relative member required")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Member escapes root")
    return path


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    temporary.replace(path)


def prepare_selection(manifests, roster_path, root, budgets, seed, allow_partial):
    if sha256_file(roster_path) != FROZEN_ROSTER_SHA:
        raise ValueError("Use the exact frozen B drive roster")
    roster = json.loads(Path(roster_path).read_text())
    by_id = {row["token"]: row for row in roster["scenes"]}
    if len(by_id) != len(roster["scenes"]):
        raise ValueError("Duplicate frozen scene")
    all_rows, seen, duplicate_inputs, snapshots = [], {}, [], []
    snapshot_root = root / "input_manifests"; snapshot_root.mkdir()
    for number, manifest_path in enumerate(manifests):
        path = Path(manifest_path).resolve(); raw = path.read_bytes(); data = json.loads(raw)
        snapshot = snapshot_root / f"manifest-{number:03d}.json"
        snapshot.write_bytes(raw)
        snapshots.append({"original_path": str(path), "snapshot": str(snapshot.relative_to(root)),
                          "sha256": hashlib.sha256(raw).hexdigest()})
        if data.get("schema") != "coordiworld.factual_only_cache.v1" or data.get("roster_sha256") != FROZEN_ROSTER_SHA:
            raise ValueError("Only full-ROI factual-cache inputs from the frozen B roster are accepted")
        config = data.get("config", {})
        if config.get("candidates", {}).get("nominal") != "pdm_closed_current_v1":
            raise ValueError("Use the fixed current PDM candidate pool")
        for record in data.get("scenes", []):
            if record.get("split") not in {"train", "validation"}:
                continue  # No calibration or confirmation arrays are opened.
            sid = record["scene_id"]
            frozen = by_id.get(sid)
            if frozen is None or frozen["role"] != record["split"] or frozen["drive"] != record.get("drive") or frozen["log"] != record.get("log_name"):
                raise ValueError("Cache identity/role differs from frozen roster")
            cache = inside(path.parent, record["path"])
            digest = record.get("cache_sha256")
            if sid in seen:
                if seen[sid] != digest:
                    raise ValueError("Incremental manifests disagree on cache bytes")
                duplicate_inputs.append(sid); continue
            seen[sid] = digest
            if sha256_file(cache) != digest:
                raise ValueError("Cache hash mismatch before input-only screening")
            scene, metadata = read_legacy_inputs(cache, record)
            if metadata.get("data_role") != frozen["role"] or metadata.get("drive_group") != frozen["drive"]:
                raise ValueError("Cached metadata drive/role differs")
            if metadata.get("truncated_agents") != 0 or metadata.get("retained_agents") != metadata.get("eligible_agents"):
                raise ValueError("S1 extension requires full current 50m ROI, not truncated actor slots")
            if record.get("factual_anchor_internal_role") not in {"fit", "probe", "development_validation"}:
                raise ValueError("Missing fixed TRAIN-fit/probe or B DEV assignment")
            if scene.candidates.shape[1:] != (12, 8, 3):
                raise ValueError("Expected fixed 12-slot, 8-step candidates")
            all_rows.append({"scene_id": sid, "token": sid, "split": frozen["role"], "roster_role": frozen["role"],
                "drive": frozen["drive"], "log_name": frozen["log"], "log": frozen["log"], "index": frozen["index"],
                "timestamp": frozen["timestamp"], "window_start": frozen["window_start"], "window_end": frozen["window_end"],
                "raw_log_sha256": frozen["sha256"], "cache_path": str(cache), "cache_sha256": digest,
                "candidate_sha256": record["candidate_sha256"],
                "factual_anchor_internal_role": record.get("factual_anchor_internal_role"),
                "geometry_screen": current_geometry_screen(scene)})
    expected = {row["token"] for row in roster["scenes"] if row["role"] in {"train", "validation"}}
    missing = sorted(expected - set(seen))
    if missing and not allow_partial:
        raise ValueError("Incomplete TRAIN/DEV input inventory; use explicit --allow-partial-input-inventory for engineering/development subset scope")
    selected, summary = select_cohorts(all_rows, budgets, seed=seed)
    if not selected:
        raise ValueError("No inputs selected")
    pilot.write_json(root / "input_screening.json", {"rows": all_rows, "no_future_or_teacher_arrays_read": True,
        "missing_roster_scene_ids": missing, "duplicate_identical_manifest_scenes": duplicate_inputs})
    return selected, summary, {"input_manifests": snapshots, "input_inventory_complete": not missing,
        "missing_roster_scenes": len(missing), "available_role_counts": dict(Counter(row["split"] for row in all_rows)),
        "scope": "partial available input inventory" if missing else "complete frozen TRAIN and B DEV input inventory"}


def initialize_worker(context):
    global WORK
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(context["map_root"]).resolve())
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from coordiworld.data.maps import CurrentMapProvider
    from coordiworld.navsim_adapter import OfficialComponentScorer
    from reactive_teacher_probe_v2 import CapturingTraffic
    geometry_audit, geometry_stats = install_corrected_builder()
    scorer = OfficialComponentScorer(context["map_root"], Path(context["output"]) / "metric_cache", "reactive")
    capture = CapturingTraffic(scorer.traffic); scorer.traffic = capture
    WORK = dict(context, map_provider=CurrentMapProvider(context["map_root"]), scorer=scorer, capture=capture,
                geometry_audit=geometry_audit, geometry_stats=geometry_stats)


def execution_rules(scene, metadata, history, execution, provider):
    from coordiworld.geometry import map_rule_margins
    p = scene.map_features.shape[1]
    tokens = provider(history, p)
    if [t["id"] for t in tokens] != [t.get("id") for t in metadata["map_token_metadata"]]:
        raise ValueError("Reconstructed current map token IDs differ")
    features = np.zeros((p, 8), dtype=np.float32); map_valid = np.zeros(p, dtype=bool)
    for i, token in enumerate(tokens):
        features[i] = token["features"]; map_valid[i] = token.get("available", True)
    if not np.allclose(features, scene.map_features[0].numpy(), atol=1e-5, rtol=0) or not np.array_equal(map_valid, scene.map_valid[0].numpy()):
        raise ValueError("Current map features/availability differ from input cache")
    margins = np.zeros(tuple(scene.rule_margins.shape[1:]), dtype=np.float32)
    valid = np.zeros_like(margins, dtype=bool)
    if tokens:
        values, masks = map_rule_margins(execution, scene.ego[0, -1, 5:7].numpy(),
                                        float(scene.ego_rear_axle_to_center[0]), tokens)
        margins[:, :, :len(tokens)], valid[:, :, :len(tokens)] = values, masks
    return margins, valid


def collect_one(row):
    from coordiworld.data.openscene import load_log
    from reactive_teacher_probe_v2 import extract_vehicle_tracks, teacher_origin
    from diagnose_idm_path import retry_dropped_endpoint
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    started = time.monotonic(); context = WORK
    context["geometry_stats"].update(calls=0, max_rear_axle_error_m=0., max_heading_error_rad=0.)
    folder = Path(context["output"]) / row["scene_id"]; folder.mkdir()
    pilot.write_json(folder / "requested_scene.json", row)
    source = Path(row["cache_path"])
    if sha256_file(source) != row["cache_sha256"]:
        raise ValueError("Selected current input cache changed")
    scene, metadata = read_legacy_inputs(source, row)
    raw = inside(context["log_root"], row["log_name"] + ".pkl")
    if sha256_file(raw) != row["raw_log_sha256"]:
        raise ValueError("Raw log differs from frozen roster")
    frames = load_log(raw); index = row["index"]
    if index < 3 or index + 8 >= len(frames):
        raise ValueError("Missing frozen history/future window")
    if (str(frames[index]["token"]) != row["scene_id"] or int(frames[index]["timestamp"]) != row["timestamp"] or
            int(frames[index - 3]["timestamp"]) != row["window_start"] or int(frames[index + 8]["timestamp"]) != row["window_end"]):
        raise ValueError("Raw log window differs from selected cache")
    history = frames[index - 3:index + 1]
    execution, full_execution, execution_receipt = predict_current_controller_execution(
        history, scene.candidates[0].numpy(), scene.candidate_valid[0].numpy())
    margins, rules = execution_rules(scene, metadata, history, execution, context["map_provider"])
    np.savez_compressed(folder / "decision_inputs.npz",
        **{"input_" + f.name: getattr(scene, f.name).numpy() for f in fields(scene)},
        executed_candidates=execution[None], executed_rule_margins=margins[None], executed_rule_valid=rules[None])
    metadata = dict(metadata, sampling_cohort=row["sampling_cohort"], geometry_screen=row["geometry_screen"])
    pilot.write_json(folder / "input_metadata.json", metadata)
    scorer, capture = context["scorer"], context["capture"]
    cache = scorer.build_metric_cache(frames, index)
    origin, coordinate_frame = teacher_origin(frames[index], np.array([cache.ego_state.rear_axle.x,
        cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading]))
    hashes = {"raw_log": row["raw_log_sha256"], "current_input_cache": row["cache_sha256"],
              "decision_inputs": sha256_file(folder / "decision_inputs.npz"),
              "input_metadata": sha256_file(folder / "input_metadata.json"),
              "collection_spec": context["collection_spec_fingerprint"]}
    initialization = pilot.initialization_record(cache, hashes, TEACHER_DOMAIN)
    init_hash = canonical_fingerprint(initialization)
    pilot.write_json(folder / "initialization.json", initialization)
    candidates = scene.candidates[0].numpy(); cv = scene.candidate_valid[0].numpy()
    eligible = ((scene.agent_class[0] == 1) & scene.agent_valid[0, -1]).numpy()
    m, n = candidates.shape[0], scene.agents.shape[2]; first = int(np.flatnonzero(cv)[0])
    states = np.zeros((m, 41, n, 7), dtype=np.float32); valid = np.zeros((m, 41, n), dtype=bool)
    scores, receipts, events = {}, [], []
    original = idm.create_path_from_se2
    idm.create_path_from_se2 = lambda poses: retry_dropped_endpoint(original, poses, events, True)
    try:
        for j in np.flatnonzero(cv):
            trial = copy.deepcopy(cache)
            if canonical_fingerprint(pilot.initialization_record(trial, hashes, TEACHER_DOMAIN)) != init_hash:
                raise ValueError("Candidate initial state differs")
            rng = pilot.reset_rng(context["seed"]); capture.calls.clear(); events.clear(); candidate_start = time.monotonic()
            score = scorer.score(trial, candidates[j])
            if not 1 <= len(capture.calls) <= 2:
                raise ValueError("Unexpected candidate/human simulation calls")
            ego, detections = capture.calls[0]
            if ego.shape != (41, 11) or len(detections) != 41:
                raise ValueError("Native simulation horizon differs")
            error = float(np.max(np.abs(ego - full_execution[int(j)])))
            if error > 1e-7:
                raise ValueError("Current-only execution differs from teacher execution")
            target, mask = extract_vehicle_tracks(detections, metadata["tracks"], eligible, origin)
            if not np.array_equal(mask[0], eligible):
                raise ValueError("Current teacher actor coverage differs")
            current_error = float(np.max(np.abs(target[0, eligible] - scene.agents[0, -1].numpy()[eligible]))) if eligible.any() else 0.
            if current_error > 1e-3:
                raise ValueError("Current teacher actor state differs")
            states[j], valid[j] = target, mask; scores[str(j)] = score
            receipt = {"candidate": int(j), "reference_candidate": first, "initialization_sha256": init_hash,
                "rng": rng, "traffic_policy": TEACHER_DOMAIN,
                "action_sha256": tensor_fingerprint(torch.from_numpy(execution[j])),
                "reference_action_sha256": tensor_fingerprint(torch.from_numpy(execution[first])),
                "current_state_max_error": current_error, "controller_state_max_error": error,
                "captured_environment_calls": len(capture.calls), "endpoint_compat_events": copy.deepcopy(events),
                "seconds": time.monotonic() - candidate_start}
            receipts.append(receipt)
            pilot.write_json(folder / f"candidate-{j:02d}.json", {"receipt": receipt, "scores": score})
            capture.calls.clear(); del trial; gc.enable(); gc.collect()
        if canonical_fingerprint(pilot.initialization_record(cache, hashes, TEACHER_DOMAIN)) != init_hash:
            raise ValueError("Base initialization mutated")
        rng = pilot.reset_rng(context["seed"]); capture.calls.clear(); events.clear()
        repeat_score = scorer.score(copy.deepcopy(cache), candidates[first])
        repeat, repeat_mask = extract_vehicle_tracks(capture.calls[0][1], metadata["tracks"], eligible, origin)
        repeated = np.array_equal(repeat, states[first]) and np.array_equal(repeat_mask, valid[first]) and repeat_score == scores[str(first)]
        if not repeated or rng != receipts[0]["rng"]:
            raise ValueError("Reference repeat is not exact under reset seed")
        repeat_ego_error = float(np.max(np.abs(capture.calls[0][0] - full_execution[first])))
        if repeat_ego_error > 1e-7:
            raise ValueError("Repeated reference execution differs")
    finally:
        idm.create_path_from_se2 = original; capture.calls.clear(); gc.enable()
    sample = np.arange(5, 41, 5); truth, mask = states[:, sample], valid[:, sample]
    common = mask & mask[first][None] & cv[:, None, None]
    delta = truth - truth[first][None]; delta[..., 2] = np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2])); delta[~common] = 0
    np.savez_compressed(folder / "teacher.npz", states_01s=states, valid_01s=valid,
        states_05s=truth, valid_05s=mask, paired_response_delta=delta, paired_response_valid=common,
        candidate_valid=cv, eligible_current_vehicle=eligible, sample_indices=sample,
        **{"simulated_ego_" + str(j): value for j, value in full_execution.items()})
    diagnostics = pilot.response_diagnostics(truth, mask, cv, first)
    receipt = {"schema": SCHEMA, "scene_id": row["scene_id"], "log_name": row["log_name"], "drive_id": row["drive"],
        "split": row["split"], "roster_role": row["roster_role"], "sampling_cohort": row["sampling_cohort"],
        "geometry_screen": row["geometry_screen"], "candidate_sha256": row["candidate_sha256"],
        "current_input_cache_sha256": row["cache_sha256"], "collection_spec_fingerprint": context["collection_spec_fingerprint"],
        "candidate_slots": m, "valid_candidates": int(cv.sum()), "current_retained_vehicles": int(eligible.sum()),
        "current_retained_agents": metadata["retained_agents"], "truncated_current_agents": 0,
        "candidate_fallback": metadata["candidate_manifest"].get("fallback"), "tracks": metadata["tracks"],
        "coordinate_frame": coordinate_frame, "initialization_sha256": init_hash, "pair_receipts": receipts,
        "execution_receipt": dict(execution_receipt, native_teacher_parity_verified=True),
        "reference_repeated_exact": True, "reference_repeat_max_position_error_m": 0.,
        "response_diagnostics": diagnostics, "teacher_domain": TEACHER_DOMAIN, "traffic_policy": TEACHER_DOMAIN,
        "idm_geometry_audit": context["geometry_audit"], "idm_geometry_runtime": dict(context["geometry_stats"]),
        "current_input_domain": "OpenScene_current_and_history_annotations_full_50m_ROI",
        "teacher_reference_domain_equals_logged_future": False,
        "physical_collision_label": None, "first_event_labels": None, "existence_labels": None,
        "native_export_ready": False, "optimizer_updates": 0, "seconds": time.monotonic() - started,
        "input_files_sha256": hashes, "teacher_npz_sha256": sha256_file(folder / "teacher.npz")}
    pilot.write_json(folder / "receipt.json", receipt)
    return receipt


def worker(row):
    remaining = WORK["deadline_epoch"] - time.time()
    if remaining <= 1:
        return {"status": "skipped", "scene_id": row["scene_id"], "reason": "total wall-clock budget"}
    def timeout(signum, frame):
        raise TimeoutError("Frozen collection wall-clock budget")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, min(WORK["max_scene_seconds"], remaining))
    try:
        return {"status": "complete", "receipt": collect_one(row)}
    except Exception as error:
        result = {"status": "failed", "scene_id": row["scene_id"], "error": repr(error), "traceback": traceback.format_exc()}
        pilot.write_json(Path(WORK["output"]) / (row["scene_id"] + ".FAILURE.json"), result)
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous); gc.enable(); gc.collect()


def reuse_scene(row, roots, destination, spec_fingerprint):
    for value in roots:
        source_root = Path(value).resolve(); report = json.loads((source_root / "report.json").read_text())
        if report.get("source_hashes_unchanged") is not True:
            raise ValueError("Prior collection lacks stable source verification")
        if row["scene_id"] not in report.get("success_scene_ids", []):
            continue
        folder = inside(source_root, row["scene_id"])
        receipt = json.loads((folder / "receipt.json").read_text())
        if (receipt.get("schema") != SCHEMA or receipt.get("collection_spec_fingerprint") != spec_fingerprint or
                receipt.get("current_input_cache_sha256") != row["cache_sha256"] or receipt.get("split") != row["split"] or
                receipt.get("sampling_cohort") != row["sampling_cohort"]):
            raise ValueError("Prior scene is incompatible with this collection plan")
        if any(path.is_symlink() for path in folder.rglob("*")):
            raise ValueError("Prior scene contains symlinks")
        if (sha256_file(folder / "teacher.npz") != receipt["teacher_npz_sha256"] or
                sha256_file(folder / "decision_inputs.npz") != receipt["input_files_sha256"]["decision_inputs"] or
                sha256_file(folder / "input_metadata.json") != receipt["input_files_sha256"]["input_metadata"] or
                receipt.get("reference_repeated_exact") is not True):
            raise ValueError("Prior collection integrity differs")
        shutil.copytree(folder, destination / row["scene_id"])
        return dict(receipt, reused_from=str(source_root))
    return None


def summarize(requested, successes, failed, skipped):
    groups = {}
    for split in ("train", "validation"):
        for cohort in ("natural", "current_geometry_enriched", "current_geometry_fallback", "ordered_engineering_pilot"):
            requested_rows = [r for r in requested if r["split"] == split and r["sampling_cohort"] == cohort]
            rows = [r for r in successes if r["split"] == split and r["sampling_cohort"] == cohort]
            ids = {r["scene_id"] for r in requested_rows}
            groups[split + "/" + cohort] = {"requested_scenes": len(requested_rows), "completed_scenes": len(rows),
                "failed_scenes": sum(r["scene_id"] in ids for r in failed),
                "skipped_scenes": sum(r["scene_id"] in ids for r in skipped),
                "completed_drives": len({r["drive_id"] for r in rows}),
                "single_valid_candidate_scenes": sum(r["valid_candidates"] < 2 for r in rows),
                "no_current_vehicle_scenes": sum(r["current_retained_vehicles"] == 0 for r in rows),
                "no_paired_cells_scenes": sum(not r["response_diagnostics"]["response_supported"] for r in rows),
                "zero_response_with_support_scenes": sum(r["response_diagnostics"]["response_supported"] and r["response_diagnostics"]["max_position_response_m"] == 0 for r in rows),
                "above_1mm_scenes": sum(r["response_diagnostics"]["above_1mm_cells"] > 0 for r in rows),
                "above_10cm_scenes": sum(r["response_diagnostics"]["above_10cm_cells"] > 0 for r in rows),
                "paired_vehicle_time_cells": sum(r["response_diagnostics"]["paired_vehicle_time_cells"] for r in rows)}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--registration", default="configs/group_cache_execution_v2.json")
    parser.add_argument("--log-root", required=True); parser.add_argument("--map-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-scenes", type=int, default=32); parser.add_argument("--dev-scenes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--selection-seed", type=int, default=20261023)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--max-seconds", type=int, default=3600); parser.add_argument("--max-scene-seconds", type=int, default=600)
    parser.add_argument("--allow-partial-input-inventory", action="store_true")
    parser.add_argument("--reuse-root", action="append", default=[])
    parser.add_argument("--compare-original-four", action="store_true", help="Use the original four preselected TRAIN identities for geometry-domain comparison; engineering only")
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 14400 or not 30 <= args.max_scene_seconds <= args.max_seconds or args.seed < 0:
        parser.error("Invalid bounded collection budget/seed")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    torch.set_num_threads(1)
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(args.map_root).resolve())
    registration = json.loads(Path(args.registration).read_text())
    dependencies = pilot.verify_runtime(registration); maps = pilot.map_receipt(args.map_root)
    geometry_audit = audit_native_builder()
    requested, sampling, inventory = prepare_selection(args.manifest, registration["roster"], root,
        {"train": args.train_scenes, "validation": args.dev_scenes}, args.selection_seed, args.allow_partial_input_inventory)
    if args.compare_original_four:
        original = pilot.select_train_scenes(json.loads(Path(registration["roster"]).read_text()), 4)
        available = {r["scene_id"]: r for r in json.loads((root / "input_screening.json").read_text())["rows"]}
        if any(r["token"] not in available for r in original):
            raise ValueError("Full-ROI cache is missing an original fixed pilot identity")
        requested = [dict(available[r["token"]], sampling_cohort="ordered_engineering_pilot") for r in original]
        sampling = {"mode": "fixed_original_four_geometry_domain_comparison", "selected_scenes": 4, "outcome_replacement": False}
    source_names = ["scripts/revision_v5_collect_paired_teachers_v2.py", "src/coordiworld/revision_v5/idm_teacher_geometry.py",
        "scripts/revision_v5_collect_paired_teachers.py", "scripts/revision_v5_collect_native_pilot.py",
        "scripts/reactive_teacher_probe_v2.py", "scripts/diagnose_idm_path.py",
        "src/coordiworld/revision_v5/native_adapter.py", "src/coordiworld/revision_v5/native_sampling.py",
        "src/coordiworld/navsim_adapter.py", "src/coordiworld/data/openscene.py", "src/coordiworld/data/maps.py", "src/coordiworld/geometry.py"]
    sources = {name: sha256_file(name) for name in source_names}
    spec = {"schema": SCHEMA, "sources": sources, "dependency_commits": registration["dependency_commits"],
            "map_manifest_sha256": maps["manifest_sha256"], "seed": args.seed, "teacher_domain": TEACHER_DOMAIN,
            "sampling_seed": args.selection_seed, "geometry_threshold_m": 2., "sampling_scope": "separate natural and enriched input cohorts"}
    spec_fingerprint = canonical_fingerprint(spec)
    plan = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "requested": requested,
            "selection": "Fixed input hash pools and drive-balanced prefixes; natural and geometry-enriched cohorts remain separate; no teacher-response selection",
            "sampling": sampling, "inventory": inventory, "collection_spec": spec, "collection_spec_fingerprint": spec_fingerprint,
            "dependencies": dependencies, "maps": maps, "idm_geometry_audit": geometry_audit,
            "comparison_original_four": args.compare_original_four, "workers": args.workers, "max_seconds": args.max_seconds,
            "max_scene_seconds": args.max_scene_seconds, "optimizer_updates": 0, "native_export_ready": False,
            "natural_and_enriched_must_not_be_pooled_as_population_rates": True}
    pilot.write_json(root / "plan.json", plan)
    successes, failures, skipped, pending = [], [], [], []
    for row in requested:
        reused = reuse_scene(row, args.reuse_root, root, spec_fingerprint)
        if reused is None: pending.append(row)
        else: successes.append(reused)
    context = {"map_root": args.map_root, "log_root": args.log_root, "output": str(root), "seed": args.seed,
        "collection_spec_fingerprint": spec_fingerprint, "max_scene_seconds": args.max_scene_seconds,
        "deadline_epoch": time.time() + max(0, args.max_seconds - (time.monotonic() - started))}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(context,)) as executor:
        futures = {executor.submit(worker, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try: result = future.result()
            except Exception as error:
                result = {"status": "failed", "scene_id": row["scene_id"], "error": repr(error)}
            if result["status"] == "complete": successes.append(result["receipt"])
            elif result["status"] == "skipped": skipped.append(result)
            else: failures.append(result)
            atomic_json(root / "status.json", {"completed": len(successes), "failed": len(failures), "skipped": len(skipped),
                "requested": len(requested), "elapsed_seconds": time.monotonic() - started})
            print(json.dumps({"scene_id": row["scene_id"], "status": result["status"], "completed": len(successes),
                "response": result.get("receipt", {}).get("response_diagnostics"), "error": result.get("error")}), flush=True)
    elapsed = time.monotonic() - started
    unchanged = all(sha256_file(name) == digest for name, digest in sources.items())
    collected = [r for r in successes if "reused_from" not in r]
    report = {"schema": SCHEMA, "success_scene_ids": [r["scene_id"] for r in successes],
        "requested_scenes": len(requested), "completed_scenes": len(successes), "failed_scenes": len(failures),
        "failures": failures, "skipped": skipped, "sampling": sampling,
        "cohort_denominators": summarize(requested, successes, failures, skipped),
        "source_hashes_unchanged": unchanged, "elapsed_seconds": elapsed,
        "newly_collected_scenes": len(collected),
        "newly_collected_scenes_per_hour": len(collected) * 3600 / max(elapsed, 1e-9),
        "candidate_simulations_this_run": sum(r["valid_candidates"] for r in collected),
        "reference_repeats_this_run": len(collected), "reused_scenes": len(successes) - len(collected),
        "artifact_candidate_count": sum(r["valid_candidates"] for r in successes),
        "optimizer_updates": 0, "native_export_ready": False, "actual_model_performance": None,
        "all_requested_completed": len(successes) == len(requested) and unchanged}
    pilot.write_json(root / "report.json", report)
    if not report["all_requested_completed"]: raise SystemExit(2)


if __name__ == "__main__":
    main()
