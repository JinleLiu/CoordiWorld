"""Input-only sampling for the prospective paired-teacher development study."""
from __future__ import annotations

import hashlib
import math

import torch

from ..geometry import rear_axle_to_box, sat_clearance


SAMPLING_DOMAIN = "revision-v5-paired-teacher-input-sampling-v1"


def order_key(value, *, seed=20261023, domain="scene"):
    return hashlib.sha256(f"{SAMPLING_DOMAIN}:{seed}:{domain}:{value}".encode()).hexdigest()


@torch.no_grad()
def current_geometry_screen(scene, *, dt=.5, margin_m=2.):
    """CV boxes use current actor state and planned ego candidates, not labels.

    This is a potential-interaction screen, neither a contact label nor a
    forecast of teacher response. Include t=0 and all valid planned candidates.
    """
    if scene.ego.shape[0] != 1 or dt <= 0 or not math.isfinite(margin_m):
        raise ValueError("Screen requires one scene, a positive dt and finite margin")
    vehicle = scene.agent_valid[0, -1] & (scene.agent_class[0] == 1)
    cv = scene.candidate_valid[0]
    m, h = scene.candidates.shape[1:3]
    result = {"definition": "minimum_OBB_SAT_clearance_current_vehicle_CV_vs_planned_valid_ego_candidates_including_t0",
              "threshold_m": float(margin_m), "dt": dt, "current_vehicle_count": int(vehicle.sum()),
              "valid_candidate_count": int(cv.sum()), "has_action_pair": bool(cv.sum() >= 2),
              "minimum_clearance_m": None, "potential_interaction": False,
              "uses_future_labels": False, "uses_teacher_response": False}
    if not vehicle.any() or not cv.any():
        return result
    dtype, device = scene.candidates.dtype, scene.candidates.device
    times = torch.arange(h + 1, dtype=dtype, device=device) * dt
    actor = scene.agents[0, -1, vehicle]
    xy = actor[None, :, :2] + times[:, None, None] * actor[None, :, 3:5]
    boxes = torch.cat((xy, actor[None, :, 2:3].expand(h + 1, -1, -1),
                       actor[None, :, 5:7].expand(h + 1, -1, -1)), -1)
    poses = torch.cat((scene.ego[0, -1, :3].expand(m, 1, 3), scene.candidates[0]), 1)[cv]
    ego = rear_axle_to_box(poses, scene.ego[0, -1, 5:7], scene.ego_rear_axle_to_center[0])
    clearance = sat_clearance(ego[:, :, None], boxes[None])
    if not torch.isfinite(clearance).all():
        raise ValueError("Nonfinite current geometry clearance")
    minimum = float(clearance.min())
    result.update(minimum_clearance_m=minimum, potential_interaction=bool(minimum <= margin_m))
    return result


def drive_balanced_order(rows, *, seed=20261023, domain="cohort"):
    groups = {}
    for row in rows:
        groups.setdefault(row["drive"], []).append(row)
    drives = sorted(groups, key=lambda drive: (order_key(drive, seed=seed, domain=domain + ":drive"), drive))
    for drive in drives:
        groups[drive].sort(key=lambda row: (order_key(row["scene_id"], seed=seed, domain=domain + ":scene"), row["scene_id"]))
    return [groups[drive][index] for index in range(max((len(group) for group in groups.values()), default=0))
            for drive in drives if index < len(groups[drive])]


def select_cohorts(rows, budgets, *, seed=20261023):
    """Each fixed hash half has a stable prefix as budgets increase.

    Natural half has no interaction filter. Enrichment half takes qualifying
    geometry first, then separately named nonqualifying input-only fallback.
    Each pool uses drive-balanced ordering. Small partial input inventories may
    exhaust a pool: keep the shortfall instead of silently changing membership.
    """
    limits = {"train": 128, "validation": 64}
    if set(budgets) != set(limits) or any(isinstance(budgets[k], bool) or not isinstance(budgets[k], int) or not 1 <= budgets[k] <= limits[k] for k in limits):
        raise ValueError("Budgets must be train=1..128 and validation=1..64")
    seen, drive_roles = set(), {}
    for row in rows:
        if row["scene_id"] in seen or row["split"] not in limits:
            raise ValueError("Duplicate scene or unsupported sampling role")
        seen.add(row["scene_id"])
        if drive_roles.setdefault(row["drive"], row["split"]) != row["split"]:
            raise ValueError("Drive occurs in both TRAIN and B DEV")
        screen = row["geometry_screen"]
        if screen.get("uses_future_labels") is not False or screen.get("uses_teacher_response") is not False:
            raise ValueError("Only declared input-only geometry may affect sampling")
    chosen, summary = [], {}
    for role, budget in budgets.items():
        pool = [row for row in rows if row["split"] == role]
        natural = [row for row in pool if int(order_key(row["scene_id"], seed=seed, domain="pool"), 16) % 2 == 0]
        enriched = [row for row in pool if int(order_key(row["scene_id"], seed=seed, domain="pool"), 16) % 2 == 1]
        interactions = [row for row in enriched if row["geometry_screen"]["potential_interaction"] and row["geometry_screen"]["has_action_pair"]]
        others = [row for row in enriched if not (row["geometry_screen"]["potential_interaction"] and row["geometry_screen"]["has_action_pair"])]
        natural_budget, enriched_budget = (budget + 1) // 2, budget // 2
        natural_order = drive_balanced_order(natural, seed=seed, domain=role + ":natural")
        interaction_order = drive_balanced_order(interactions, seed=seed, domain=role + ":enriched")
        fallback_order = drive_balanced_order(others, seed=seed, domain=role + ":fallback")
        selected = [dict(row, sampling_cohort="natural") for row in natural_order[:natural_budget]]
        selected += [dict(row, sampling_cohort="current_geometry_enriched") for row in interaction_order[:enriched_budget]]
        fallback_budget = max(0, enriched_budget - len(interaction_order))
        selected += [dict(row, sampling_cohort="current_geometry_fallback") for row in fallback_order[:fallback_budget]]
        chosen.extend(selected)
        summary[role] = {"requested_budget": budget, "selected_scenes": len(selected),
            "selected_drives": len({row["drive"] for row in selected}), "input_available_scenes": len(pool),
            "natural_pool": len(natural), "enrichment_pool": len(enriched), "interaction_pool": len(interactions),
            "shortfall": budget - len(selected),
            "cohort_counts": {name: sum(row["sampling_cohort"] == name for row in selected)
                              for name in ("natural", "current_geometry_enriched", "current_geometry_fallback")}}
    return chosen, summary
