from pathlib import Path

import pytest

from anomdet.datasets.inventory import discover_dataset


def _config(root: Path) -> dict:
    return {
        "capture": {"supported_protocols": ["dns"]},
        "data": {
            "dataset_root": str(root),
            "benign_pcap_dir": "benign/pcap",
            "attack_pcap_dir": "attack/pcap",
            "attack_label_dir": "attack/labels",
            "capture_allowlist": {
                "benign": {"dns": ["small.pcap"]},
                "attack": {"dns": ["attack.pcap"]},
            },
        },
    }


def test_inventory_respects_explicit_capture_allowlist(tmp_path: Path) -> None:
    for relative in [
        "benign/pcap/dns/small.pcap",
        "benign/pcap/dns/large.pcap",
        "attack/pcap/dns/attack.pcap",
        "attack/pcap/dns/other.pcap",
        "attack/labels/dns/attack.csv",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    inventory = discover_dataset(_config(tmp_path))

    assert [Path(row["path"]).name for row in inventory["benign"]] == ["small.pcap"]
    assert [Path(row["capture"]).name for row in inventory["attack_pairs"]] == ["attack.pcap"]


def test_inventory_rejects_unknown_allowlist_capture(tmp_path: Path) -> None:
    available = tmp_path / "benign/pcap/dns/available.pcap"
    available.parent.mkdir(parents=True)
    available.touch()
    config = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="do not exist"):
        discover_dataset(config)
