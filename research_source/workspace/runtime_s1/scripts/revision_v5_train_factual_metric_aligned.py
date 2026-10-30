"""Metric-aligned factual development revision on frozen B TRAIN only.

This entry point never qualifies an anchor, uses risk/value labels, or evaluates
NAVSIM. A small/partial cache is accepted only with --engineering-only. Longer
runs use a new output directory and a preregistered epoch budget; no automatic
resume, checkpoint search, or promotion is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from coordiworld.model import CoordiWorld, ModelConfig
from coordiworld.schema import SceneInputs


CHECKPOINT_STATUS = "unqualified_factual_pretraining"
B_ROSTER_SHA256 = "9b930a7da8ac2bf8a4012544c66400b6d5788aa30bc5ac96772937484fd6ce6d"
LOSS_WEIGHTS = {"xy_euclidean_ADE": 1.0}
OBJECTIVE_VERSION = "factual_xy_ade_after_internal_probe_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inside(root, member):
    relative = Path(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Cache paths must remain relative to the manifest")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("Cache path escapes manifest root")
    return target


def _order(seed, domain, value):
    return hashlib.sha256(f"factual-anchor-v1:{seed}:{domain}:{value}".encode()).hexdigest()


def make_plan(manifest_path, roster_path, *, split_seed=20261023, probe_drives=7,
              engineering_only=False, max_fit_scenes=None, max_probe_scenes=None):
    """Assign roles using identity metadata only, before opening any NPZ."""
    manifest_path, roster_path = Path(manifest_path), Path(roster_path)
    manifest, roster = _read(manifest_path), _read(roster_path)
    roster_digest = sha256_file(roster_path)
    if not engineering_only and roster_digest != B_ROSTER_SHA256:
        raise ValueError("Nonengineering pretraining requires the exact frozen B roster SHA256")
    if ("scenes" in manifest) == ("records" in manifest):
        raise ValueError("Provide exactly one manifest scenes or records list")
    raw_records = manifest.get("scenes", manifest.get("records"))
    if not isinstance(raw_records, list) or not isinstance(roster.get("scenes"), list):
        raise ValueError("Scene and roster records must be lists")
    if probe_drives < 1:
        raise ValueError("At least one probe drive is required")
    if any(n is not None and n < 1 for n in (max_fit_scenes, max_probe_scenes)):
        raise ValueError("Scene limits must be positive")
    if not engineering_only and (max_fit_scenes is not None or max_probe_scenes is not None):
        raise ValueError("Scene-limited runs require --engineering-only")
    roster_by_id = {}
    drive_roles = defaultdict(set)
    for row in roster["scenes"]:
        token, drive, role = row.get("token"), row.get("drive"), row.get("role")
        if not all(isinstance(v, str) and v for v in (token, drive, role)) or token in roster_by_id:
            raise ValueError("Missing or duplicate roster identity")
        roster_by_id[token] = row
        drive_roles[drive].add(role)
    if any(len(roles) != 1 for roles in drive_roles.values()):
        raise ValueError("A frozen roster drive appears in multiple roles")
    expected = {token for token, row in roster_by_id.items() if row["role"] == "train"}
    selected, seen, seen_paths, ignored = [], set(), set(), 0
    for original in raw_records:
        sid = original.get("scene_id", original.get("token"))
        if not isinstance(sid, str) or sid in seen:
            raise ValueError("Missing or duplicate cache scene identity")
        seen.add(sid)
        role = original.get("split", original.get("role"))
        if role != "train":
            if sid in expected:
                raise ValueError("B TRAIN scene has a conflicting cache role")
            ignored += 1
            continue
        if sid not in expected:
            raise ValueError("Cache TRAIN scene is not in frozen B TRAIN")
        r = roster_by_id[sid]
        if original.get("drive", original.get("drive_group")) != r["drive"]:
            raise ValueError("Cache drive differs from frozen roster")
        if original.get("log_name", original.get("log")) != r["log"]:
            raise ValueError("Cache log differs from frozen roster")
        digest = original.get("cache_sha256", original.get("sha256"))
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "study_scene_000027" for c in digest):
            raise ValueError("Each cache requires a SHA256 identity")
        path = inside(manifest_path.parent, original["path"])
        if path in seen_paths:
            raise ValueError("Multiple scenes point to the same cache file")
        seen_paths.add(path)
        selected.append({"scene_id": sid, "drive": r["drive"], "log_name": r["log"],
                         "path": original["path"], "cache_sha256": digest,
                         "timestamp": r["timestamp"], "roster_role": "train"})
    available = {row["scene_id"] for row in selected}
    if not engineering_only and available != expected:
        raise ValueError("Incomplete B TRAIN cache requires --engineering-only")
    available_drives = {row["drive"] for row in selected}
    all_train_drives = {roster_by_id[token]["drive"] for token in expected}
    assignment_pool = available_drives if engineering_only else all_train_drives
    if len(assignment_pool) <= probe_drives:
        raise ValueError("Need more available TRAIN drives than probe_drives")
    ordered = sorted(assignment_pool, key=lambda d: (_order(split_seed, "drive", d), d))
    probe_set, fit_set = set(ordered[:probe_drives]), set(ordered[probe_drives:])
    roles = {}
    for role, drives, limit in (("fit", fit_set, max_fit_scenes),
                                ("probe", probe_set, max_probe_scenes)):
        rows = sorted((row for row in selected if row["drive"] in drives),
                      key=lambda row: (_order(split_seed, "scene", row["scene_id"]), row["scene_id"]))
        roles[role] = rows if limit is None else rows[:limit]
        if not roles[role]:
            raise ValueError("Both fit and probe must have at least one scene")
    plan = {"schema": "coordiworld.factual_anchor_plan.v1", "manifest_sha256": sha256_file(manifest_path),
            "roster_sha256": roster_digest, "matches_frozen_B_roster": roster_digest == B_ROSTER_SHA256,
            "split_seed": split_seed,
            "probe_drive_budget": probe_drives, "engineering_only": engineering_only,
            "assignment_scope": "available_TRAIN_drives_engineering_only" if engineering_only else "all_frozen_B_TRAIN_drives",
            "drive_order_rule": "SHA256(factual-anchor-v1:<split_seed>:drive:<drive>), first probe_drives are probe",
            "assigned_fit_drives": sorted(fit_set), "assigned_probe_drives": sorted(probe_set),
            "records": roles, "expected_B_TRAIN_scenes": len(expected), "available_B_TRAIN_scenes": len(available),
            "missing_B_TRAIN_scene_ids": sorted(expected - available), "ignored_nontrain_manifest_records": ignored,
            "scene_limits": {"fit": max_fit_scenes, "probe": max_probe_scenes},
            "manifest_requested_count": manifest.get("requested_count"),
            "manifest_failed": manifest.get("failed", manifest.get("failures")),
            "manifest_skipped": manifest.get("skipped"), "inventory_complete": False,
            "probe_interpretation": "Internal development probe from B TRAIN, not independent final confirmation",
            "checkpoint_status": CHECKPOINT_STATUS, "qualification_receipt_created": False,
            "teacher_reference_domain_alignment_verified": False}
    plan["fingerprint"] = canonical_sha(plan)
    return plan, manifest


def load_sample(root, row):
    """Read only current input fields and logged factual future; never risk labels."""
    path = inside(root, row["path"])
    if sha256_file(path) != row["cache_sha256"]:
        raise ValueError("Cache content SHA256 differs")
    with np.load(path, allow_pickle=False) as archive:
        scene = SceneInputs(**{f.name: torch.from_numpy(archive["input_" + f.name].copy()) for f in fields(SceneInputs)})
        future = torch.from_numpy(archive["target_future_agents"].copy())
        valid = torch.from_numpy(archive["target_future_valid"].copy())
        metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("scene_id") != row["scene_id"] or metadata.get("log_name") != row["log_name"]:
        raise ValueError("Cache metadata scene/log identity differs")
    if metadata.get("data_role", "train") != "train" or metadata.get("drive_group", row["drive"]) != row["drive"]:
        raise ValueError("Cache metadata TRAIN drive differs")
    if metadata.get("timestamp") != row["timestamp"] or metadata.get("input_latest_timestamp") != row["timestamp"]:
        raise ValueError("Input timestamp differs from frozen current timestamp")
    scene = replace(scene, **{f.name: getattr(scene, f.name).float() for f in fields(scene)
                              if getattr(scene, f.name).is_floating_point()})
    scene.validate()
    b, _, n, _ = scene.agents.shape
    h = scene.candidates.shape[2]
    if b != 1 or future.shape != (1, h, n, 7) or valid.shape != (1, h, n) or valid.dtype != torch.bool:
        raise ValueError("Expected one scene with logged future [1,H,N,7] and bool mask")
    if not future.is_floating_point() or not torch.isfinite(future[valid]).all():
        raise ValueError("Available logged targets must be finite floating point")
    if (future[..., 5:7][valid] <= 0).any():
        raise ValueError("Available logged actor sizes must be positive")
    # Invalid targets may be NaN: clear them before subtraction, not afterwards.
    future = torch.where(valid[..., None], future.float(), 0)
    return scene, future, valid


def factual_inputs(scene):
    """Remove planned/logged ego trajectories from the action-independent path."""
    b, _, h, _ = scene.candidates.shape
    p = scene.map_features.shape[1]
    return replace(scene, candidates=scene.candidates.new_zeros((b, 1, h, 3)),
                   candidate_valid=torch.ones((b, 1), dtype=torch.bool, device=scene.ego.device),
                   rule_margins=scene.rule_margins.new_zeros((b, 1, h, p, 4)),
                   rule_valid=torch.zeros((b, 1, h, p, 4), dtype=torch.bool, device=scene.ego.device))


def factual_loss(output, scene, future, valid):
    """Scene mean XY Euclidean displacement on observed current-actor support.

    No variance, yaw, velocity, existence, risk or value target enters this
    objective. Missing future observations are never negative existence labels.
    """
    mask = valid & scene.agent_valid[:, -1, None]
    predicted = output["states"][:, 0]
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Non-finite factual prediction")
    distance = torch.linalg.vector_norm(predicted[..., :2] - future[..., :2], dim=-1)
    loss = distance[mask].mean() if mask.any() else predicted.sum() * 0
    return loss, {"xy_euclidean_ADE": loss}


def constant_velocity_xy(scene, horizon, dt):
    current = scene.agents[:, -1]
    times = torch.arange(1, horizon + 1, dtype=current.dtype, device=current.device) * dt
    return current[:, None, :, :2] + times[None, :, None, None] * current[:, None, :, 3:5]


def initialize_model(config, seed, device):
    if config.action_conditioned:
        raise ValueError("Factual anchor must be action_conditioned=False")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = CoordiWorld(config).to(device=device, dtype=torch.float32)
    if config.motion_prior != "constant_velocity":
        raise ValueError("This preregistered factual initializer requires the constant-velocity prior")
    # Explicit new-run initialization, matching the existing core constructor.
    # No historical checkpoint or development outcome determines these values.
    with torch.no_grad():
        model.motion_head[-1].weight.zero_(); model.motion_head[-1].bias.zero_()
        model.motion_head[-1].bias[5] = 3.0
    # These branches are never called or trained. Their random weights stay in
    # the legacy state_dict solely for strict downstream checkpoint loading.
    for name, parameter in model.named_parameters():
        if (name.startswith(("collision_head.", "action_embedding.")) or
                ".action." in name or ".norms.0." in name):
            parameter.requires_grad_(False)
    output_head = model.motion_head[-1]
    model._frozen_unsupervised_heads = (output_head.weight[5:].detach().clone(), output_head.bias[5:].detach().clone())
    def no_unsupervised_gradient(gradient):
        result = gradient.clone(); result[5:] = 0
        return result
    output_head.weight.register_hook(no_unsupervised_gradient)
    output_head.bias.register_hook(no_unsupervised_gradient)
    return model


@torch.no_grad()
def restore_frozen_unsupervised(model):
    """Keep unsupervised existence and variance outputs exactly initialized."""
    model.motion_head[-1].weight[5:].copy_(model._frozen_unsupervised_heads[0])
    model.motion_head[-1].bias[5:].copy_(model._frozen_unsupervised_heads[1])


@torch.no_grad()
def evaluate(model, rows, root, device):
    model.eval(); scene_rows = []
    for row in rows:
        scene, future, valid = load_sample(root, row)
        scene, future, valid = scene.to(device), future.to(device), valid.to(device)
        output = model.rollout(factual_inputs(scene))
        loss, parts = factual_loss(output, scene, future, valid)
        mask = valid & scene.agent_valid[:, -1, None]
        error = torch.linalg.vector_norm(output["states"][:, 0, ..., :2] - future[..., :2], dim=-1)
        cv_error = torch.linalg.vector_norm(constant_velocity_xy(scene, future.shape[1], model.config.dt) - future[..., :2], dim=-1)
        support = int(mask.sum()); value = float(error[mask].mean()) if support else None
        cv = float(cv_error[mask].mean()) if support else None
        scene_rows.append({"scene_id": row["scene_id"], "drive": row["drive"], "actor_time_support": support,
                           "current_actors": int(scene.agent_valid[:, -1].sum()), "ADE_m": value,
                           "CV_ADE_m": cv, "ADE_minus_CV_m": value - cv if support else None,
                           "factual_loss": float(loss), "loss_parts": {k: float(v) for k, v in parts.items()}})
    supported = [r for r in scene_rows if r["actor_time_support"]]
    drive = defaultdict(list)
    for row in supported:
        drive[row["drive"]].append(row)
    drive_rows = [{"drive": name, "supported_scenes": len(group),
                   "ADE_m": sum(r["ADE_m"] for r in group) / len(group),
                   "CV_ADE_m": sum(r["CV_ADE_m"] for r in group) / len(group)} for name, group in sorted(drive.items())]
    support = sum(r["actor_time_support"] for r in supported)
    avg = lambda key: sum(r[key] for r in supported) / len(supported) if supported else None
    drive_avg = lambda key: sum(r[key] for r in drive_rows) / len(drive_rows) if drive_rows else None
    mean, cv = drive_avg("ADE_m"), drive_avg("CV_ADE_m")
    return {"requested_scenes": len(rows), "supported_scenes": len(supported),
            "unsupported_scenes": len(rows) - len(supported), "supported_drives": len(drive_rows),
            "actor_time_support": support, "scene_equal_ADE_m": avg("ADE_m"), "scene_equal_CV_ADE_m": avg("CV_ADE_m"),
            "drive_equal_ADE_m": mean, "drive_equal_CV_ADE_m": cv,
            "drive_equal_ADE_minus_CV_m": mean - cv if mean is not None else None,
            "relative_ADE_reduction_vs_CV": (cv - mean) / cv if cv is not None and cv > 0 else None,
            "actor_time_pooled_ADE_m": sum(r["ADE_m"] * r["actor_time_support"] for r in supported) / support if support else None,
            "scene_rows": scene_rows, "drive_rows": drive_rows,
            "uncertainty_calibration_verified": False, "independent_confirmation": False}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    names = ["scripts/revision_v5_train_factual_metric_aligned.py", "src/coordiworld/__init__.py",
             "src/coordiworld/model.py", "src/coordiworld/schema.py", "src/coordiworld/geometry.py"]
    return {name: sha256_file(root / name) for name in names}


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if not 1 <= args.epochs <= 100 or not 0 < args.learning_rate <= .01 or args.accumulate_scenes < 1:
        raise ValueError("Invalid epoch, learning-rate, or accumulation budget")
    if args.weight_decay != 0 or args.cpu_threads < 1:
        raise ValueError("Invalid weight decay or CPU thread count")
    if (args.epochs != 10 or args.learning_rate != 1e-4 or args.accumulate_scenes != 8 or
            args.seed not in (17, 29, 43) or args.split_seed != 20261023 or args.probe_drives != 7 or
            args.engineering_only or args.max_fit_scenes is not None or args.max_probe_scenes is not None):
        raise ValueError("Fixed revision budget: full TRAIN, 10 epochs, LR1e-4, accum8, seeds17/29/43, fixed48/7")
    plan, manifest = make_plan(args.manifest, args.roster, split_seed=args.split_seed,
                              probe_drives=args.probe_drives, engineering_only=args.engineering_only,
                              max_fit_scenes=args.max_fit_scenes, max_probe_scenes=args.max_probe_scenes)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("Requested CUDA unavailable or unsupported device; no silent fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=False)
    root = Path(args.manifest).parent
    registration = {"plan": plan, "seed": args.seed, "epochs": args.epochs, "precision": "float32",
                    "device_requested": args.device, "device_used": str(device), "torch_version": str(torch.__version__),
                    "numpy_version": str(np.__version__), "source_sha256": source_fingerprints(),
                    "objective_version": OBJECTIVE_VERSION,
                    "development_revision_after_internal_probe_read": True,
                    "revision_reason": "Prior three-seed composite-loss factual runs underperformed CV on internal TRAIN probe; align position objective with Euclidean ADE",
                    "prior_internal_probe_all_entity_ADE_m_by_seed": {"17": .607438, "29": .582520, "43": .614699},
                    "prior_internal_probe_all_entity_CV_ADE_m": .566771,
                    "prior_internal_probe_moving_vehicle_ADE_m_by_seed": {"17": 2.229119, "29": 2.220200, "43": 2.229945},
                    "prior_internal_probe_moving_vehicle_CV_ADE_m": 2.224400,
                    "B_DEV_outcomes_read_for_this_revision": False,
                    "loss_aggregation": "observed actor-time Euclidean XY distance mean per scene, scene-equal gradient accumulation; reporting remains drive-equal",
                    "random_initialization_only": True, "checkpoint_policy": "fixed_terminal_epoch_no_probe_checkpoint_selection",
                    "initialization": "seeded random trunk; explicit zero final motion output weights/bias except frozen existence bias=3; constant_velocity prior",
                    "epoch_zero_CV_ADE_parity_tolerance_m": 0.0001,
                    "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                    "accumulate_supported_scenes": args.accumulate_scenes, "loss_weights": LOSS_WEIGHTS,
                    "target_fields_read": ["target_future_agents", "target_future_valid"],
                    "future_valid_semantics": "observed label support, never physical existence truth",
                    "existence_output_frozen": True, "variance_outputs_frozen": True, "existence_probability_interpretation": "untrained constant output; not calibrated physical probability",
                    "risk_or_value_supervision_used": False, "logged_ego_used_as_input": False,
                    "checkpoint_status": CHECKPOINT_STATUS, "qualification_receipt_created": False,
                    "teacher_reference_domain_alignment_verified": False, "official_EPDMS": None}
    # This identity-only split and requested budget are persisted before the
    # first cache tensor is read; later capacity inference cannot change roles.
    _save(output / "plan.json", registration)
    try:
        # Capacity/horizon are determined from selected current inputs, not errors.
        capacity, history, horizons = 1, 1, set()
        for row in plan["records"]["fit"] + plan["records"]["probe"]:
            scene, _, _ = load_sample(root, row)
            capacity = max(capacity, scene.agents.shape[2]); history = max(history, scene.ego.shape[1])
            horizons.add(scene.candidates.shape[2])
        if len(horizons) != 1:
            raise ValueError("Do not mix factual horizons")
        config_values = _read(args.model_config) if args.model_config else {}
        if config_values.get("action_conditioned", False):
            raise ValueError("Factual anchor requires action_conditioned=False")
        config_values.setdefault("max_agents", capacity); config_values.setdefault("max_history", max(8, history))
        config_values.setdefault("dropout", 0.0); config_values["action_conditioned"] = False
        config = ModelConfig(**config_values)
        if config.max_agents < capacity or config.max_history < history:
            raise ValueError("Model capacity is smaller than selected current inputs")
        data_dt = manifest.get("config", {}).get("data", {}).get("dt", .5)
        if config.dt != data_dt:
            raise ValueError("Model dt differs from cache sampling")
        registration["model_config"] = asdict(config)
        registration["fingerprint"] = canonical_sha(registration)
        _save(output / "registration.json", registration)
        model = initialize_model(config, args.seed, device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.learning_rate, weight_decay=args.weight_decay)
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = {role: evaluate(model, rows, root, device) for role, rows in plan["records"].items()}
        _save(output / "epoch-000-metrics.json", initial)
        if not initial["fit"]["actor_time_support"] or not initial["probe"]["actor_time_support"]:
            raise ValueError("Both fit and probe need factual position support; no invented ADE")
        if any(abs(row["ADE_minus_CV_m"]) > registration["epoch_zero_CV_ADE_parity_tolerance_m"]
               for role in initial.values() for row in role["scene_rows"] if row["actor_time_support"]):
            raise ValueError("Epoch-zero motion must equal CV within the registered FP32 tolerance")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rng = random.Random(args.seed); updates = 0; start = time.monotonic()
        with (output / "epochs.jsonl").open("x") as log:
            for epoch in range(1, args.epochs + 1):
                _sync(device); epoch_start = time.monotonic(); model.train(); order = list(plan["records"]["fit"]); rng.shuffle(order)
                optimizer.zero_grad(set_to_none=True); pending = 0; loss_sum = 0.; supported_count = 0; skipped = 0
                def step():
                    nonlocal pending, updates
                    if not pending:
                        return
                    for p in parameters:
                        if p.grad is not None:
                            p.grad.div_(pending)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    if float(norm) > 0:
                        optimizer.step(); restore_frozen_unsupervised(model); updates += 1
                    optimizer.zero_grad(set_to_none=True); pending = 0
                for row in order:
                    scene, future, valid = load_sample(root, row)
                    scene, future, valid = scene.to(device), future.to(device), valid.to(device)
                    if not (valid & scene.agent_valid[:, -1, None]).any():
                        skipped += 1
                        continue
                    loss, _ = factual_loss(model.rollout(factual_inputs(scene)), scene, future, valid)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite factual training loss")
                    loss.backward(); pending += 1; supported_count += 1; loss_sum += float(loss.detach())
                    if pending == args.accumulate_scenes:
                        step()
                step(); _sync(device); training_seconds = time.monotonic() - epoch_start
                metrics = {role: evaluate(model, rows, root, device) for role, rows in plan["records"].items()}
                _sync(device)
                _save(output / f"epoch-{epoch:03d}-metrics.json", metrics)
                record = {"epoch": epoch, "optimizer_updates_cumulative": updates,
                          "fit_requested_scenes": len(order), "fit_optimization_scenes": supported_count,
                          "no_observed_actor_future_scenes_retained_without_update": skipped,
                          "online_fit_loss": loss_sum / supported_count if supported_count else None,
                          "training_seconds_including_cache_reads": training_seconds,
                          "fit_scenes_per_second": len(order) / training_seconds,
                          "epoch_seconds_including_evaluation": time.monotonic() - epoch_start,
                          "probe_drive_equal_ADE_m": metrics["probe"]["drive_equal_ADE_m"],
                          "probe_drive_equal_CV_ADE_m": metrics["probe"]["drive_equal_CV_ADE_m"],
                          "checkpoint_status": CHECKPOINT_STATUS}
                log.write(json.dumps(record, allow_nan=False) + "\n"); log.flush()
                print(json.dumps(record, allow_nan=False), flush=True)
        if (not torch.equal(model.motion_head[-1].weight[5:], model._frozen_unsupervised_heads[0]) or
                not torch.equal(model.motion_head[-1].bias[5:], model._frozen_unsupervised_heads[1])):
            raise RuntimeError("Unsupervised existence/variance output changed")
        checkpoint = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "model_config": asdict(config), "model_name": "coordiworld", "seed": args.seed,
                      "epoch": args.epochs, "optimizer_updates": updates, "checkpoint_status": CHECKPOINT_STATUS,
                      "provenance": registration, "action_conditioned": False,
                      "risk_head_trained": False, "value_head_trained": False,
                      "existence_head_trained": False, "variance_heads_trained": False, "future_valid_is_physical_existence": False,
                      "objective_version": OBJECTIVE_VERSION, "development_revision_after_internal_probe_read": True,
                      "uncertainty_calibration_verified": False, "teacher_domain_alignment_verified": False}
        torch.save(checkpoint, output / "factual_anchor.pt")
        restored = torch.load(output / "factual_anchor.pt", map_location="cpu", weights_only=True)
        if not all(torch.equal(v, restored["model_state"][k]) for k, v in checkpoint["model_state"].items()):
            raise RuntimeError("Checkpoint roundtrip mismatch")
        result = {"status": "completed_unqualified", "checkpoint_status": CHECKPOINT_STATUS,
                  "engineering_only": args.engineering_only, "registration_sha256": sha256_file(output / "registration.json"),
                  "checkpoint_sha256": sha256_file(output / "factual_anchor.pt"), "epochs": args.epochs, "objective_version": OBJECTIVE_VERSION,
                  "development_revision_after_internal_probe_read": True, "optimizer_updates": updates, "device_used": str(device),
                  "elapsed_seconds_including_evaluation": time.monotonic() - start,
                  "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "GPU_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  "final_probe_drive_equal_ADE_m": metrics["probe"]["drive_equal_ADE_m"],
                  "final_probe_drive_equal_CV_ADE_m": metrics["probe"]["drive_equal_CV_ADE_m"],
                  "qualification_receipt_created": False, "teacher_reference_domain_alignment_verified": False,
                  "source_sha256_unchanged": source_fingerprints() == registration["source_sha256"],
                  "official_EPDMS": None, "independent_confirmation": False}
        if not result["source_sha256_unchanged"]:
            raise RuntimeError("Source changed during training")
        _save(output / "result.json", result)
        return result
    except Exception as error:
        _save(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                        "checkpoint_status": CHECKPOINT_STATUS,
                                        "qualification_receipt_created": False})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--roster", required=True)
    p.add_argument("--output", required=True); p.add_argument("--model-config", help="JSON ModelConfig fields")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--split-seed", type=int, default=20261023); p.add_argument("--probe-drives", type=int, default=7)
    p.add_argument("--engineering-only", action="store_true")
    p.add_argument("--max-fit-scenes", type=int); p.add_argument("--max-probe-scenes", type=int)
    p.add_argument("--device", default="auto", help="auto uses CUDA when available; explicit cuda never silently falls back")
    p.add_argument("--cpu-threads", type=int, default=4); p.add_argument("--accumulate-scenes", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.)
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
