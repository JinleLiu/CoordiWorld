"""Versioned tensor archive with separate decision inputs and supervision.

No pickle objects are loaded. Declarations cannot prove source authenticity:
native exporter and controller parity still require server integration tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from ..schema import SceneInputs
from .contracts import (DatasetManifest, ExposureRecord, LabelSemantics,
                        PairedResponseRecord, validate_manifest, validate_paired_responses)
from .losses import RevisionV5Targets


@dataclass
class RevisionBatch:
    scene: SceneInputs
    executed_candidates: torch.Tensor
    reference_action: torch.Tensor
    executed_rule_margins: torch.Tensor
    executed_rule_valid: torch.Tensor
    targets: RevisionV5Targets
    manifest: DatasetManifest
    data_origin: str
    pair_records: tuple[PairedResponseRecord, ...] = ()

    def validate(self, exposure_history=()):
        summary = validate_manifest(self.manifest, exposure_history)
        b = self.scene.ego.shape[0]
        if b != len(self.manifest.records):
            raise ValueError("Manifest rows and tensor scene count differ")
        if self.data_origin not in {"synthetic_engineering_fixture", "native_export"}:
            raise ValueError("Unknown data origin")
        if self.executed_candidates.shape != self.scene.candidates.shape:
            raise ValueError("Executed candidate shape mismatch")
        if self.reference_action.shape != (b,) or self.reference_action.dtype != torch.long:
            raise ValueError("One long reference candidate index per scene is required")
        if self.executed_rule_margins.shape != self.scene.rule_margins.shape or self.executed_rule_valid.shape != self.executed_rule_margins.shape:
            raise ValueError("Execution-specific rule geometry mismatch")
        required = {f.name for f in fields(SceneInputs)} | {
            "executed_candidates", "reference_action", "executed_rule_margins", "executed_rule_valid"}
        if not required.issubset(self.manifest.input_availability):
            raise ValueError("Every decision input requires an availability declaration")
        mask = self.targets.target_valid & self.scene.candidate_valid
        if self.targets.response_supervision_valid is not None:
            mask &= self.targets.response_supervision_valid
        if self.data_origin == "native_export" and mask.any():
            validate_paired_responses(self.pair_records)
            records = {r.pair_id: r for r in self.pair_records}
            expected = set()
            for i, row in enumerate(self.manifest.records):
                ref = int(self.reference_action[i])
                if not 0 <= ref < mask.shape[1] or not self.scene.candidate_valid[i, ref]:
                    raise ValueError("Invalid reference candidate")
                for j in range(mask.shape[1]):
                    if not mask[i, j]:
                        continue
                    key = f"{row.scene_id}:{j}"
                    expected.add(key)
                    r = records.get(key)
                    if r is None or r.scene_id != row.scene_id:
                        raise ValueError("Missing candidate response provenance: " + key)
                    if (r.simulator_domain != self.targets.domain_id or
                        r.traffic_policy != self.manifest.semantics.traffic_policy or
                        r.initial_state_sha256 != self.targets.paired_initial_state_ids[i][j] or
                        str(r.random_seed) != self.targets.paired_rng_ids[i][j]):
                        raise ValueError("Response provenance differs from tensor metadata")
                    if r.action_sha256 != tensor_fingerprint(self.executed_candidates[i, j]) or r.reference_action_sha256 != tensor_fingerprint(self.executed_candidates[i, ref]):
                        raise ValueError("Response provenance action differs from decision execution")
            if set(records) != expected:
                raise ValueError("Unexpected response provenance rows")
        return summary

    def to(self, device):
        return RevisionBatch(self.scene.to(device), self.executed_candidates.to(device),
                             self.reference_action.to(device), self.executed_rule_margins.to(device),
                             self.executed_rule_valid.to(device), self.targets.to(device),
                             self.manifest, self.data_origin, self.pair_records)


def tensor_fingerprint(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    prefix = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(prefix + b"\n" + array.tobytes()).hexdigest()


def save_batch(batch: RevisionBatch, directory) -> Path:
    batch.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    arrays, target_metadata = {}, {}
    for f in fields(SceneInputs):
        arrays["scene__" + f.name] = getattr(batch.scene, f.name).detach().cpu().numpy()
    for name in ("executed_candidates", "reference_action", "executed_rule_margins", "executed_rule_valid"):
        arrays["decision__" + name] = getattr(batch, name).detach().cpu().numpy()
    for f in fields(RevisionV5Targets):
        value = getattr(batch.targets, f.name)
        if isinstance(value, torch.Tensor):
            arrays["target__" + f.name] = value.detach().cpu().numpy()
        else:
            target_metadata[f.name] = value
    path = root / "tensors.npz"
    np.savez_compressed(path, **arrays)
    metadata = {"format": "coordiworld_revision_v5_batch_v1", "data_origin": batch.data_origin,
                "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manifest": asdict(batch.manifest), "target_metadata": target_metadata,
                "pair_records": [asdict(x) for x in batch.pair_records]}
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return root


def load_batch(directory, *, exposure_history=()) -> RevisionBatch:
    root = Path(directory)
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata["format"] != "coordiworld_revision_v5_batch_v1":
        raise ValueError("Unsupported batch format")
    path = root / "tensors.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["tensor_sha256"]:
        raise ValueError("Tensor archive hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    scene = SceneInputs(**{f.name: values.pop("scene__" + f.name) for f in fields(SceneInputs)})
    decision = {name: values.pop("decision__" + name) for name in
                ("executed_candidates", "reference_action", "executed_rule_margins", "executed_rule_valid")}
    target = dict(metadata["target_metadata"])
    allowed = {f.name for f in fields(RevisionV5Targets)}
    for key, value in values.items():
        if not key.startswith("target__") or key[8:] not in allowed or key[8:] in target:
            raise ValueError("Unknown or duplicated target array: " + key)
        target[key[8:]] = value
    for name in ("paired_initial_state_ids", "paired_rng_ids"):
        target[name] = tuple(tuple(row) for row in target[name])
    manifest = dict(metadata["manifest"])
    manifest["semantics"] = LabelSemantics(**manifest["semantics"])
    manifest["records"] = tuple(ExposureRecord(**x) for x in manifest["records"])
    batch = RevisionBatch(scene=scene, targets=RevisionV5Targets(**target),
                          manifest=DatasetManifest(**manifest), data_origin=metadata["data_origin"],
                          pair_records=tuple(PairedResponseRecord(**x) for x in metadata["pair_records"]), **decision)
    batch.validate(exposure_history)
    return batch
