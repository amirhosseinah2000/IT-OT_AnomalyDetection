"""Discover the repository's protocol-first benign and attack data layout.

The discovery layer deliberately depends on folders and evidence, rather than a
hand-written list of every capture.  A deployment can therefore replace the
dataset root without changing feature extraction or model code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomdet.core.io import utc_now, write_json
from anomdet.core.paths import resolve_capture_paths


@dataclass(frozen=True)
class CaptureSource:
    """One PCAP source with an explicit split and protocol."""

    split: str
    protocol: str
    path: Path

    def as_dict(self) -> dict[str, str]:
        return {"split": self.split, "protocol": self.protocol, "path": str(self.path)}


@dataclass(frozen=True)
class AttackPair:
    """One attack PCAP and its label CSV, including pairing evidence."""

    protocol: str
    capture: Path
    labels: Path
    pairing_method: str
    pairing_score: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "protocol": self.protocol,
            "capture": str(self.capture),
            "labels": str(self.labels),
            "pairing_method": self.pairing_method,
            "pairing_score": self.pairing_score,
        }


def _tokens(path: Path) -> set[str]:
    """Return stable semantic tokens while ignoring export and protocol suffixes."""
    stem = path.name.lower()
    stem = re.sub(r"\.(pcapng|pcap|csv)$", "", stem)
    stem = stem.replace(".pcap_flow", "")
    parts = re.split(r"[^a-z0-9]+", stem)
    ignored = {
        "pcap",
        "flow",
        "dns",
        "http",
        "mb",
        "fast",
        "attack",
        "label",
        "labels",
        "csv",
    }
    return {part for part in parts if len(part) > 1 and part not in ignored}


def _name_similarity(capture: Path, labels: Path) -> float:
    left, right = _tokens(capture), _tokens(labels)
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _manual_pairs(config: dict[str, Any], protocol: str) -> dict[str, str]:
    """Read optional explicit pair overrides without coupling the rest of the code to YAML."""
    declared = config.get("data", {}).get("attack_pair_overrides", {})
    values = declared.get(protocol, {}) if isinstance(declared, dict) else {}
    return values if isinstance(values, dict) else {}


def _pair_protocol(
    config: dict[str, Any], protocol: str, pcap_dir: Path, label_dir: Path
) -> list[AttackPair]:
    captures = resolve_capture_paths(pcap_dir)
    labels = sorted(label_dir.glob("*.csv"), key=lambda path: path.name.casefold())
    if not labels:
        return []
    overrides = _manual_pairs(config, protocol)
    assigned: set[Path] = set()
    pairs: list[AttackPair] = []
    for capture in captures:
        if capture.name in overrides:
            target = label_dir / overrides[capture.name]
            if not target.exists():
                raise FileNotFoundError(
                    f"Attack-pair override for {capture.name} points to missing CSV: {target}"
                )
            assigned.add(target)
            pairs.append(AttackPair(protocol, capture, target, "configured_override", 1.0))
            continue
        candidates = [(candidate, _name_similarity(capture, candidate)) for candidate in labels]
        candidate, score = max(candidates, key=lambda item: (item[1], item[0].name.casefold()))
        if score < 0.45 or candidate in assigned:
            # Keep this pair discoverable but mark it for an evidence-based mapping decision.
            pairs.append(AttackPair(protocol, capture, candidate, "needs_mapping_evidence", score))
        else:
            assigned.add(candidate)
            pairs.append(AttackPair(protocol, capture, candidate, "name_similarity", score))
    return pairs


def discover_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Discover benign captures and attack PCAP/CSV pairs from the portable layout.

    Required layout::

        data/raw/
          benign/pcap/<protocol>/*.{pcap,pcapng}
          attack/pcap/<protocol>/*.{pcap,pcapng}
          attack/labels/<protocol>/*.csv

    Pairing by filename is only a candidate.  The mapping stage later verifies
    the pair using shared endpoint and timestamp evidence before labels are used.
    """
    data = config.get("data", {})
    root = Path(data.get("dataset_root", "data/raw"))
    supported = [str(item).lower() for item in config["capture"]["supported_protocols"]]
    benign_root = root / str(data.get("benign_pcap_dir", "benign/pcap"))
    attack_pcap_root = root / str(data.get("attack_pcap_dir", "attack/pcap"))
    attack_label_root = root / str(data.get("attack_label_dir", "attack/labels"))
    benign: list[CaptureSource] = []
    attack_captures: list[CaptureSource] = []
    attack_pairs: list[AttackPair] = []
    for protocol in supported:
        benign_dir = benign_root / protocol
        if benign_dir.is_dir():
            benign.extend(
                CaptureSource("benign", protocol, path)
                for path in resolve_capture_paths(benign_dir)
            )
        attack_dir, label_dir = attack_pcap_root / protocol, attack_label_root / protocol
        if attack_dir.is_dir():
            attack_captures.extend(
                CaptureSource("attack", protocol, path)
                for path in resolve_capture_paths(attack_dir)
            )
        if attack_dir.is_dir() and label_dir.is_dir():
            attack_pairs.extend(_pair_protocol(config, protocol, attack_dir, label_dir))
    return {
        "schema_version": "2.0.0",
        "created_at": utc_now(),
        "dataset_root": str(root),
        "benign": [item.as_dict() for item in benign],
        "attack_captures": [item.as_dict() for item in attack_captures],
        "attack_pairs": [item.as_dict() for item in attack_pairs],
        "counts": {
            "benign_captures": len(benign),
            "attack_captures": len(attack_captures),
            "attack_pairs": len(attack_pairs),
        },
    }


def write_dataset_inventory(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Persist the discovered layout so external systems can inspect the exact inputs."""
    inventory = discover_dataset(config)
    write_json(inventory, output_path)
    return inventory
