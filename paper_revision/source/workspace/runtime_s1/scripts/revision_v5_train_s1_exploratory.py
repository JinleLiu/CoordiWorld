"""Bounded S1 teacher-response learning, always unqualified_exploratory.

This isolated entry point accepts no checkpoint and never calls the formal
training_step or alters its exposure gate. Historical B/F completeness remains
false. Reference pretraining and response fitting use the same frozen TRAIN-fit
drives. Internal TRAIN-probe and B DEV are development reports only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from coordiworld.geometry import rear_axle_to_box, sat_clearance, wrap_angle
from coordiworld.schema import SceneInputs
from coordiworld.revision_v5.contracts import canonical_fingerprint
from coordiworld.revision_v5.data import tensor_fingerprint
from coordiworld.revision_v5.losses import RevisionV5LossConfig, RevisionV5Targets, revision_v5_loss
from coordiworld.revision_v5.model import RevisionV5Config, RevisionV5Model, _clean_scene, _reference_indices


STATUS = "unqualified_exploratory"
FORMAT = "coordiworld_s1_exploratory_only_v1"
ROSTER_SHA = "9b930a7da8ac2bf8a4012544c66400b6d5788aa30bc5ac96772937484fd6ce6d"
SPLIT_SEED = 20261023
RELATIVE_DENOMINATOR_FLOOR_M = .05
INTERACTION_CLEARANCE_M = 2.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inside(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path must remain inside the collection root")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes collection root")
    return path


def train_drive_roles(roster):
    roles, ids = {}, set()
    for row in roster["scenes"]:
        if row["token"] in ids or roles.setdefault(row["drive"], row["role"]) != row["role"]:
            raise ValueError("Duplicate scene or cross-role drive in frozen roster")
        ids.add(row["token"])
    drives = sorted((d for d, r in roles.items() if r == "train"), key=lambda d: (
        hashlib.sha256(f"factual-anchor-v1:{SPLIT_SEED}:drive:{d}".encode()).hexdigest(), d))
    if len(drives) <= 7:
        raise ValueError("Frozen TRAIN roster must contain more than seven drives")
    return {d: "probe" if i < 7 else "fit" for i, d in enumerate(drives)}


def make_plan(roots, roster_path):
    """Freeze identities and file hashes before opening any tensor archive."""
    roster_path = Path(roster_path)
    if sha(roster_path) != ROSTER_SHA:
        raise ValueError("S1 requires the exact frozen B roster SHA256")
    roster = read(roster_path); assignments = train_drive_roles(roster)
    lookup = {r["token"]: r for r in roster["scenes"]}
    result, seen, collections = [], set(), []
    for supplied in roots:
        root = Path(supplied).resolve()
        report, plan = read(root / "report.json"), read(root / "plan.json")
        if report.get("source_hashes_unchanged") is not True:
            raise ValueError("Collector sources changed or lack a source stability receipt")
        requested = plan.get("requested", plan.get("requested_scenes"))
        if not isinstance(requested, list):
            raise ValueError("Collector plan requires the predeclared requested identity list")
        request_ids = {r.get("token", r.get("scene_id")): r for r in requested}
        if len(request_ids) != len(requested):
            raise ValueError("Duplicate requested collector scene")
        successes = report.get("success_scene_ids")
        if not isinstance(successes, list) or len(successes) != len(set(successes)):
            raise ValueError("Collector success IDs must be unique")
        requested_denominators = defaultdict(int)
        for sid, requested_row in request_ids.items():
            if sid not in lookup or lookup[sid]["role"] not in {"train", "validation"}:
                raise ValueError("Requested collection contains unknown/CAL/confirmation identities")
            group = str(requested_row.get("sampling_cohort", "ordered_engineering_pilot"))
            requested_denominators[lookup[sid]["role"] + ":" + group] += 1
        collections.append({"root": str(root), "plan_sha256": sha(root / "plan.json"),
            "report_sha256": sha(root / "report.json"), "requested_count": len(requested),
            "completed_count": len(successes), "requested_role_cohort_counts": dict(requested_denominators),
            "failures": report.get("failures", []),
            "skipped": report.get("skipped", []), "selection": plan.get("selection"),
            "all_requested_completed": report.get("all_requested_completed", report.get("integration_checks_passed_for_all_requested"))})
        for sid in successes:
            if not isinstance(sid, str) or sid in seen or sid not in request_ids or sid not in lookup:
                raise ValueError("Unknown, duplicate or undeclared scene identity")
            seen.add(sid); frozen = lookup[sid]; requested_row = request_ids[sid]
            role = frozen["role"]
            if role not in {"train", "validation"}:
                raise ValueError("S1 must never open CAL or confirmation data")
            current = inside(root, sid)
            metadata, receipt = read(current / "input_metadata.json"), read(current / "receipt.json")
            if (receipt.get("scene_id") != sid or receipt.get("drive_id") != frozen["drive"] or
                    receipt.get("log_name") != frozen["log"]):
                raise ValueError("Collector receipt identity differs from frozen roster")
            allowed = {"train"} if role == "train" else {"validation", "development"}
            if receipt.get("split") not in allowed or metadata.get("data_role") not in allowed:
                raise ValueError("Collector role differs from frozen roster")
            if (metadata.get("scene_id") != sid or metadata.get("log_name") != frozen["log"] or
                    metadata.get("drive_group") != frozen["drive"] or
                    metadata.get("timestamp") != frozen["timestamp"] or
                    metadata.get("input_latest_timestamp") != frozen["timestamp"]):
                raise ValueError("Current-input identity/timestamp differs from frozen roster")
            internal = assignments[frozen["drive"]] if role == "train" else "development_validation"
            for document in (requested_row, metadata):
                if document.get("factual_anchor_internal_role", internal) != internal:
                    raise ValueError("TRAIN fit/probe assignment differs from frozen full-drive hash")
            names = ["decision_inputs.npz", "teacher.npz", "input_metadata.json", "receipt.json"]
            hashes = {name: sha(current / name) for name in names}
            if (hashes["teacher.npz"] != receipt.get("teacher_npz_sha256") or
                    hashes["decision_inputs.npz"] != receipt.get("input_files_sha256", {}).get("decision_inputs")):
                raise ValueError("Collector array SHA256 mismatch")
            cohort = requested_row.get("sampling_cohort", requested_row.get("selection_stratum", "ordered_engineering_pilot"))
            if any(document.get("sampling_cohort", cohort) != cohort for document in (metadata, receipt)):
                raise ValueError("Natural/enrichment cohort differs across collection records")
            result.append({"scene_id": sid, "drive": frozen["drive"], "log_name": frozen["log"],
                "timestamp": frozen["timestamp"], "root": str(current), "roster_role": role,
                "role": internal, "sampling_cohort": cohort, "files_sha256": hashes})
    result.sort(key=lambda row: (row["role"], row["drive"], row["scene_id"]))
    if not any(row["role"] == "fit" for row in result):
        raise ValueError("No completed TRAIN-fit scenes; no replacement or role reassignment")
    role_denominators = {role: {"completed_scenes": sum(r["role"] == role for r in result),
        "completed_drives": len({r["drive"] for r in result if r["role"] == role})}
        for role in ("fit", "probe", "development_validation")}
    plan = {"format": FORMAT, "status": STATUS, "inventory_complete": False,
        "current_dependency_scope_complete": True, "old_or_public_learned_dependencies": [],
        "initialization": "random weights with explicit zero motion/response residual priors",
        "roster_sha256": ROSTER_SHA, "collections": collections, "records": result,
        "actual_role_denominators": role_denominators,
        "TRAIN_drive_roles": assignments, "split_seed": SPLIT_SEED,
        "probe_rule": "first7 of SHA256(factual-anchor-v1:20261023:drive:<drive>), all55 TRAIN drives",
        "selection_policy": "fixed identity assignments; failures and zero responses retained in denominators",
        "development_only": True, "confirmation_read": False,
        "interaction_condition": "current vehicle CV vs PLANNED candidate OBB SAT clearance <=2m, t=0..4s",
        "relative_denominator_floor_m": RELATIVE_DENOMINATOR_FLOOR_M,
        "relative_floor_basis": "design stability threshold fixed after TRAIN engineering pilot, before B DEV outcomes; not measured noise or a physical law",
        "low_baseline_interpretation": "Report absolute differences; meaningful small absolute improvements remain possible"}
    plan["fingerprint"] = canonical_fingerprint(plan)
    return plan


def load_sample(row):
    root = Path(row["root"])
    if any(sha(root / name) != value for name, value in row["files_sha256"].items()):
        raise ValueError("Frozen sample SHA256 changed")
    receipt = read(root / "receipt.json")
    with np.load(root / "decision_inputs.npz", allow_pickle=False) as data:
        scene = SceneInputs(**{f.name: torch.from_numpy(data["input_" + f.name].copy()) for f in fields(SceneInputs)})
        execution = torch.from_numpy(data["executed_candidates"].copy())
        margins = torch.from_numpy(data["executed_rule_margins"].copy())
        rules = torch.from_numpy(data["executed_rule_valid"].copy())
    scene = replace(scene, **{f.name: getattr(scene, f.name).float() for f in fields(scene)
        if getattr(scene, f.name).is_floating_point()})
    execution, margins = execution.float(), margins.float()
    scene.validate(); _clean_scene(scene, execution, margins, rules)
    if scene.ego.shape[0] != 1 or scene.candidates.shape[2] != 8:
        raise ValueError("Expected one current scene and eight future 0.5s steps")
    with np.load(root / "teacher.npz", allow_pickle=False) as data:
        truth = torch.from_numpy(data["states_05s"].copy()).float()[None]
        valid = torch.from_numpy(data["valid_05s"].copy())[None]
        candidate_valid = torch.from_numpy(data["candidate_valid"].copy())[None]
        eligible = torch.from_numpy(data["eligible_current_vehicle"].copy())[None]
        saved_delta = torch.from_numpy(data["paired_response_delta"].copy()).float()[None]
        saved_common = torch.from_numpy(data["paired_response_valid"].copy())[None]
        indices = data["sample_indices"].copy()
    b, m, h, _ = scene.candidates.shape; n = scene.agents.shape[2]
    if (truth.shape != (1, m, h, n, 7) or valid.shape != (1, m, h, n) or
            valid.dtype != torch.bool or candidate_valid.dtype != torch.bool or
            not torch.equal(candidate_valid, scene.candidate_valid) or
            not torch.equal(eligible, (scene.agent_class == 1) & scene.agent_valid[:, -1]) or
            eligible.dtype != torch.bool or not np.array_equal(indices, np.arange(5, 41, 5))):
        raise ValueError("Teacher/current actor, candidate, sampling or shape mismatch")
    if (valid & ~eligible[:, None, None]).any() or not torch.isfinite(truth[valid]).all():
        raise ValueError("Teacher validity includes unavailable actor or nonfinite target")
    references = {r["reference_candidate"] for r in receipt["pair_receipts"]}
    if len(references) != 1:
        raise ValueError("Inconsistent reference action")
    reference = next(iter(references)); _reference_indices(reference, candidate_valid)
    if reference != int(torch.nonzero(candidate_valid[0])[0]):
        raise ValueError("Collector first-valid reference differs")
    common = valid & valid[:, reference:reference + 1] & candidate_valid[..., None, None]
    delta = truth - truth[:, reference:reference + 1]
    delta[..., 2] = wrap_angle(delta[..., 2]); delta = torch.where(common[..., None], delta, 0)
    if (saved_delta.shape != delta.shape or saved_common.dtype != torch.bool or
            not torch.equal(common, saved_common) or not torch.allclose(delta, saved_delta, atol=1e-5, rtol=1e-5)):
        raise ValueError("Stored paired response differs from paired state differences")
    pairs = receipt["pair_receipts"]
    if sorted(p["candidate"] for p in pairs) != torch.nonzero(candidate_valid[0]).flatten().tolist():
        raise ValueError("Paired receipts must cover each valid candidate exactly once")
    rng_ids, init_ids, domains = [], [], []
    for pair in pairs:
        rng = pair["rng"]; clean_rng = {k: v for k, v in rng.items() if k != "fingerprint"}
        if canonical_fingerprint(clean_rng) != rng.get("fingerprint"):
            raise ValueError("RNG receipt fingerprint differs")
        if (pair["initialization_sha256"] != receipt["initialization_sha256"] or
                pair["action_sha256"] != tensor_fingerprint(execution[0, pair["candidate"]]) or
                pair["reference_action_sha256"] != tensor_fingerprint(execution[0, reference]) or
                not 0 <= pair["controller_state_max_error"] <= 1e-7 or
                not 0 <= pair["current_state_max_error"] <= 1e-3):
            raise ValueError("Paired initialization/action/controller parity receipt differs")
        rng_ids.append(rng["fingerprint"]); init_ids.append(pair["initialization_sha256"])
        domains.append(pair["traffic_policy"])
    if (len(set(rng_ids)) != 1 or len(set(init_ids)) != 1 or len(set(domains)) != 1 or
            domains[0] != receipt.get("traffic_policy", receipt.get("teacher_domain")) or not receipt.get("reference_repeated_exact") or
            receipt.get("execution_receipt", {}).get("native_teacher_parity_verified") is not True):
        raise ValueError("Same-domain paired initial state/RNG/repeat is not verified")
    # Unavailable teacher values may be NaN; never impute them as supervision.
    truth = torch.where(valid[..., None], truth, 0)
    targets = RevisionV5Targets(truth, valid, candidate_valid, domains[0], domains[0],
        (tuple([init_ids[0]] * m),), (tuple([rng_ids[0]] * m),), response_supervision_valid=candidate_valid)
    return {"scene": scene, "execution": execution, "margins": margins, "rules": rules,
            "reference": reference, "targets": targets}


def to_device(sample, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in sample.items()}


def world_forward(model, sample):
    """Reuse V5 world and loss, without executing untrained risk/value branches."""
    scene = sample["scene"]
    x = _clean_scene(scene, sample["execution"], sample["margins"], sample["rules"])
    if x.agents.shape[2] > model.config.max_agents or x.ego.shape[1] > model.config.max_history:
        raise ValueError("Scene exceeds preregistered model capacity")
    reference = _reference_indices(sample["reference"], x.candidate_valid)
    world, _, _ = model._world(x, reference, model._action_features(x))
    b, m, h, _, _ = world["states"].shape
    zero = world["states"].sum() * 0
    return dict(world, candidate_valid=x.candidate_valid, reference_action=reference,
        actor_valid=x.agent_valid[:, -1], hazard=zero.expand(b, m, h, 2),
        raw_event_cdf=zero.expand(b, m, h, 2), utility=zero.expand(b, m), progress=zero.expand(b, m))


def planned_interaction_actors(scene, dt=.5):
    """Input-only potential interaction; not a collision or response label."""
    b, m, h, _ = scene.candidates.shape
    current = scene.agents[:, -1]; n = current.shape[1]
    t = torch.arange(h + 1, device=current.device, dtype=current.dtype) * dt
    actor = current[:, None].expand(-1, h + 1, -1, -1).clone()
    actor[..., :2] += t[None, :, None, None] * current[:, None, :, 3:5]
    initial = scene.ego[:, -1, :3][:, None, None].expand(-1, m, 1, -1)
    plans = torch.cat((initial, scene.candidates), 2)
    ego_box = rear_axle_to_box(plans, scene.ego[:, -1, 5:7][:, None, None],
                              scene.ego_rear_axle_to_center[:, None, None])
    actor_box = torch.cat((actor[..., :3], actor[..., 5:7]), -1)
    clearance = sat_clearance(ego_box[..., None, :], actor_box[:, None])
    eligible = (scene.agent_class == 1) & scene.agent_valid[:, -1]
    return ((clearance <= INTERACTION_CLEARANCE_M) & scene.candidate_valid[:, :, None, None]).any(2).any(1) & eligible


def initialize(config, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = RevisionV5Model(config)
    # Preserve the existing CV prior, including the deliberately untrained
    # existence prior; no learned weights are read by this executable.
    with torch.no_grad():
        model.anchor.motion_head[-1].weight[:5].zero_()
        model.anchor.motion_head[-1].bias[:5].zero_()
    return model


def configure_stage(model, stage):
    model.requires_grad_(False)
    if stage == "reference":
        model.config.freeze_anchor = False
        model.anchor.requires_grad_(True)
        for name, parameter in model.anchor.named_parameters():
            if name.startswith(("collision_head.", "action_embedding.")) or ".action." in name or ".norms.0." in name:
                parameter.requires_grad_(False)
    elif stage == "response":
        model.config.freeze_anchor = True
        for module in (model.response_action, model.response_temporal, model.response_head):
            module.requires_grad_(True)
    else:
        raise ValueError("Unknown S1 stage")
    return [p for p in model.parameters() if p.requires_grad]


def loss(output, sample):
    return revision_v5_loss(output, sample["targets"], RevisionV5LossConfig(
        response_weight=1., factual_weight=0., existence_weight=0., risk_weight=0.,
        hazard_weight=0., utility_weight=0., progress_weight=0., consistency_weight=0., pairwise_weight=0.))


def metric_row(output, sample, row, *, subgroup="all"):
    targets = sample["targets"]; ref = sample["reference"]
    mask = targets.paired_valid & targets.paired_valid[:, ref:ref + 1]
    mask &= targets.target_valid[..., None, None] & output["actor_valid"][:, None, None]
    mask[:, ref] = False
    if subgroup not in {"all", "interaction", "noninteraction"}: raise ValueError("Unknown diagnostic subgroup")
    if subgroup != "all":
        geometry = planned_interaction_actors(sample["scene"])
        mask &= (geometry if subgroup == "interaction" else ~geometry)[:, None, None]
    truth_delta = targets.paired_states[..., :2] - targets.paired_states[:, ref:ref + 1, ..., :2]
    error = torch.linalg.vector_norm(output["response_delta"][..., :2] - truth_delta, dim=-1)
    baseline = torch.linalg.vector_norm(truth_delta, dim=-1)
    count = int(mask.sum()); average = lambda value: float(value[mask].mean()) if count else None
    factual_mask = targets.paired_valid[:, ref] & output["actor_valid"][:, None]
    factual_error = torch.linalg.vector_norm(output["anchor_states"][..., :2] - targets.paired_states[:, ref, ..., :2], dim=-1)
    return {"scene_id": row["scene_id"], "drive": row["drive"], "sampling_cohort": row["sampling_cohort"],
        "paired_actor_time_cells": count, "delta_ADE_m": average(error), "zero_delta_ADE_m": average(baseline),
        "paired_cells_above_10cm": int((mask & (baseline > .1)).sum()),
        "teacher_reference_ADE_m": float(factual_error[factual_mask].mean()) if factual_mask.any() else None,
        "reference_response_max_abs": float(output["response_delta"][:, ref].abs().max()) if output["response_delta"][:, ref].numel() else 0.}


def aggregate(rows):
    supported = [r for r in rows if r["paired_actor_time_cells"]]
    drives = defaultdict(list)
    for row in supported: drives[row["drive"]].append(row)
    drive_rows = [{"drive": d, **{k: sum(r[k] for r in rs) / len(rs)
                    for k in ("delta_ADE_m", "zero_delta_ADE_m")}} for d, rs in sorted(drives.items())]
    mean = lambda k: sum(r[k] for r in drive_rows) / len(drive_rows) if drive_rows else None
    estimate, baseline = mean("delta_ADE_m"), mean("zero_delta_ADE_m")
    reference_drives = defaultdict(list)
    for row in rows:
        if row.get("teacher_reference_ADE_m") is not None:
            reference_drives[row["drive"]].append(row["teacher_reference_ADE_m"])
    reference_means = [sum(v) / len(v) for v in reference_drives.values()]
    return {"requested_scenes": len(rows), "supported_scenes": len(supported), "supported_drives": len(drives),
        "paired_actor_time_cells": sum(r["paired_actor_time_cells"] for r in rows),
        "paired_cells_above_10cm": sum(r["paired_cells_above_10cm"] for r in rows),
        "drive_equal_delta_ADE_m": estimate, "drive_equal_zero_delta_ADE_m": baseline,
        "drive_equal_teacher_reference_ADE_m": sum(reference_means) / len(reference_means) if reference_means else None,
        "absolute_improvement_m": baseline - estimate if estimate is not None else None,
        "relative_improvement": (baseline - estimate) / baseline if baseline is not None and baseline >= RELATIVE_DENOMINATOR_FLOOR_M else None,
        "relative_denominator_floor_m": RELATIVE_DENOMINATOR_FLOOR_M,
        "relative_not_reported_reason": "no supported paired cells" if baseline is None else (
            "below preregistered design stability floor, not evidence of zero effect" if baseline < RELATIVE_DENOMINATOR_FLOOR_M else None),
        "relative_floor_is_measured_noise": False,
        "drive_rows": drive_rows, "scene_rows": rows, "independent_confirmation": False}


@torch.no_grad()
def evaluate(model, samples, records):
    model.eval(); grouped = defaultdict(list)
    for sample, row in zip(samples, records):
        output = world_forward(model, sample)
        natural = metric_row(output, sample, row)
        interaction = metric_row(output, sample, row, subgroup="interaction")
        noninteraction = metric_row(output, sample, row, subgroup="noninteraction")
        grouped[(row["role"], "all_selected")].append(natural)
        grouped[(row["role"], "current_geometry_interaction_actors")].append(interaction)
        grouped[(row["role"], "current_geometry_noninteraction_actors")].append(noninteraction)
        grouped[(row["role"], "selection_" + str(row["sampling_cohort"]))].append(natural)
        if row["sampling_cohort"] == "natural":
            grouped[(row["role"], "natural_current_geometry_interaction_actors")].append(interaction)
            grouped[(row["role"], "natural_current_geometry_noninteraction_actors")].append(noninteraction)
    return {role: {name: aggregate(rows) for (r, name), rows in grouped.items() if r == role}
            for role in sorted({key[0] for key in grouped})}


@torch.no_grad()
def verify_cv_epoch_zero(model, samples):
    model.eval(); error = 0.
    for sample in samples:
        output = world_forward(model, sample); scene = sample["scene"]
        h = scene.candidates.shape[2]
        times = torch.arange(1, h + 1, device=scene.ego.device, dtype=scene.ego.dtype) * model.config.dt
        current = scene.agents[:, -1]
        cv = current[:, None, :, :2] + times[None, :, None, None] * current[:, None, :, 3:5]
        mask = scene.agent_valid[:, -1, None].expand(-1, h, -1)
        if mask.any(): error = max(error, float((output["anchor_states"][..., :2] - cv)[mask].abs().max()))
        if output["response_delta"].ne(0).any(): raise ValueError("Epoch-zero response must be identically zero")
    if error > 1e-4: raise ValueError("Epoch-zero anchor differs from CV prior")
    return {"maximum_xy_component_error_m": error, "tolerance_m": 1e-4, "response_delta_exactly_zero": True}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ["scripts/revision_v5_train_s1_exploratory.py", "src/coordiworld/model.py", "src/coordiworld/schema.py",
             "src/coordiworld/geometry.py", "src/coordiworld/revision_v5/model.py", "src/coordiworld/revision_v5/losses.py",
             "src/coordiworld/revision_v5/contracts.py", "src/coordiworld/revision_v5/data.py"]
    return {name: sha(root / name) for name in names}


def run(args):
    started = time.monotonic()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("Use a new output directory")
    if not all(1 <= n <= 100 for n in (args.anchor_epochs, args.response_epochs)):
        raise ValueError("Each fixed epoch budget must be 1..100")
    if not 0 < args.learning_rate <= .01 or args.cpu_threads < 1 or args.accumulate_scenes < 1:
        raise ValueError("Invalid training budget")
    if not 60 <= args.max_seconds <= 14400 or args.seed < 0:
        raise ValueError("Invalid wall-clock budget or seed")
    plan = make_plan(args.pilot_root, args.roster)
    output.mkdir(parents=True, exist_ok=False)
    registration = {"format": FORMAT, "status": STATUS, "plan": plan, "seed": args.seed,
        "anchor_epochs": args.anchor_epochs, "response_epochs": args.response_epochs,
        "max_seconds": args.max_seconds,
        "learning_rate": args.learning_rate, "optimizer": "Adam, weight_decay=0; no decay of unsupervised outputs",
        "accumulate_scenes": args.accumulate_scenes, "checkpoint_policy": "fixed terminal epoch only",
        "old_or_public_checkpoint_inputs": [], "source_sha256": source_hashes(), "precision": "float32",
        "teacher": "paired privileged IDM response, no real-world causal interpretation",
        "qualification_receipt_created": False, "inventory_complete": False}
    save(output / "registration.json", registration)
    try:
        def check_deadline():
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("Fixed S1 wall-clock budget exhausted; no automatic retry or partial promotion")
        torch.set_num_threads(args.cpu_threads); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device(args.device)
        if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device is unavailable; no fallback")
        samples = []
        for row in plan["records"]:
            check_deadline(); samples.append(to_device(load_sample(row), device))
        values = read(args.model_config) if args.model_config else {}
        protected = {"freeze_anchor": False, "dropout": 0., "action_conditioned": True, "risk_world_mode": "predicted",
                     "dt": .5, "use_provenance": True, "use_source": True, "use_lineage": True,
                     "use_ambiguity": True, "use_confidence": True}
        if any(k in values and values[k] != v for k, v in protected.items()):
            raise ValueError("S1 model config overrides a fixed mechanism contract")
        values.update(protected)
        values.setdefault("max_agents", max(s["scene"].agents.shape[2] for s in samples))
        values.setdefault("max_history", max(8, max(s["scene"].ego.shape[1] for s in samples)))
        config = RevisionV5Config(**values); model = initialize(config, args.seed).to(device)
        save(output / "model_config.json", asdict(config))
        save(output / "epoch_zero_cv_check.json", verify_cv_epoch_zero(model, samples))
        save(output / "epoch_zero_metrics.json", evaluate(model, samples, plan["records"]))
        fit_indices = [i for i, r in enumerate(plan["records"]) if r["role"] == "fit"]
        updates = {"reference": 0, "response": 0}; start = time.monotonic()
        for stage, epochs in (("reference", args.anchor_epochs), ("response", args.response_epochs)):
            params = configure_stage(model, stage)
            optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.)
            rng = random.Random(args.seed + (0 if stage == "reference" else 1000003))
            # Existence/variance have no labels here; their output rows must not move.
            unsupervised = (model.anchor.motion_head[-1].weight[5:].detach().clone(),
                            model.anchor.motion_head[-1].bias[5:].detach().clone())
            frozen_anchor = {k: v.detach().cpu().clone() for k, v in model.anchor.state_dict().items()} if stage == "response" else None
            with (output / (stage + "_epochs.jsonl")).open("x") as stream:
                for epoch in range(1, epochs + 1):
                    model.train(); order = list(fit_indices); rng.shuffle(order); optimizer.zero_grad(set_to_none=True)
                    total, supported, pending = 0., 0, 0
                    def step():
                        nonlocal pending
                        if not pending: return
                        for p in params:
                            if p.grad is not None: p.grad.div_(pending)
                        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0; updates[stage] += 1
                    for index in order:
                        check_deadline()
                        sample = samples[index]; out = world_forward(model, sample); losses = loss(out, sample)
                        name, count = ("factual", "factual_supervised_elements") if stage == "reference" else ("response", "response_supervised_elements")
                        if not int(losses[count]): continue
                        objective = losses[name]
                        if not torch.isfinite(objective): raise FloatingPointError("Nonfinite S1 loss")
                        objective.backward(); pending += 1; supported += 1; total += float(objective.detach())
                        if pending == args.accumulate_scenes: step()
                    step()
                    if not supported: raise ValueError("No supervised TRAIN-fit support; no success inferred")
                    record = {"stage": stage, "epoch": epoch, "fit_scenes_requested": len(order),
                        "fit_scenes_supported": supported, "mean_scene_objective": total / supported,
                        "optimizer_updates": updates[stage], "status": STATUS}
                    stream.write(json.dumps(record) + "\n"); stream.flush(); print(json.dumps(record), flush=True)
            if not torch.equal(unsupervised[0], model.anchor.motion_head[-1].weight[5:]) or not torch.equal(unsupervised[1], model.anchor.motion_head[-1].bias[5:]):
                raise RuntimeError("Unsupervised existence/variance output weights changed")
            if frozen_anchor is not None and any(not torch.equal(v, model.anchor.state_dict()[k].detach().cpu()) for k, v in frozen_anchor.items()):
                raise RuntimeError("Anchor changed during response stage")
            save(output / (stage + "_terminal_metrics.json"), evaluate(model, samples, plan["records"]))
        if registration["source_sha256"] != source_hashes(): raise RuntimeError("Sources changed during run")
        if sha(args.roster) != plan["roster_sha256"] or any(
                sha(Path(c["root"]) / filename) != c[field]
                for c in plan["collections"]
                for filename, field in (("plan.json", "plan_sha256"), ("report.json", "report_sha256"))):
            raise RuntimeError("Frozen collection plan/report/roster changed during run")
        if any(sha(Path(r["root"]) / name) != digest for r in plan["records"] for name, digest in r["files_sha256"].items()):
            raise RuntimeError("Sample evidence changed during run")
        checkpoint = {"format": FORMAT, "status": STATUS, "exploratory_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "exploratory_config": asdict(model.config), "registration_sha256": sha(output / "registration.json"),
            "qualified": False, "inventory_complete": False, "learned_from_old_checkpoint": False,
            "risk_head_trained": False, "value_head_trained": False, "existence_supervised": False,
            "teacher_reference_domain_qualified": False}
        torch.save(checkpoint, output / "unqualified_exploratory.pt")
        result = {"status": STATUS, "completed": True, "optimizer_updates": updates,
            "actual_role_denominators": plan["actual_role_denominators"],
            "checkpoint_sha256": sha(output / "unqualified_exploratory.pt"), "inventory_complete": False,
            "official_EPDMS": None, "qualification_receipt_created": False, "independent_confirmation": False,
            "seed": args.seed, "device": str(device), "elapsed_seconds": time.monotonic() - start,
            "risk_or_value_labels_read": False, "source_and_array_hashes_unchanged": True}
        save(output / "result.json", result); return result
    except Exception as error:
        save(output / "failure.json", {"status": STATUS, "error_type": type(error).__name__, "error": str(error)})
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-root", action="append", required=True)
    p.add_argument("--roster", required=True); p.add_argument("--output", required=True)
    p.add_argument("--model-config"); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--anchor-epochs", type=int, default=20); p.add_argument("--response-epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--accumulate-scenes", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--cpu-threads", type=int, default=4); p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
