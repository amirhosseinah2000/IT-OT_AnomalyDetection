"""User-friendly input-path resolution for PCAP and PCAPNG command arguments."""

from __future__ import annotations

from pathlib import Path

PCAP_SUFFIXES = {".pcap", ".pcapng"}


def resolve_capture_path(requested: Path) -> Path:
    """Resolve an existing capture and recover common `.pcap.pcapng` naming mistakes."""
    if requested.is_file():
        return requested
    parent = requested.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Capture directory does not exist: {parent}")
    names = {requested.name.casefold(), f"{requested.name}.pcapng".casefold()}
    if requested.suffix.casefold() == ".pcap":
        names.add(requested.with_suffix(".pcapng").name.casefold())
    matches = sorted(
        [
            candidate
            for candidate in parent.iterdir()
            if candidate.is_file() and candidate.name.casefold() in names
        ],
        key=lambda candidate: candidate.name.casefold(),
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        choices = ", ".join(str(match) for match in matches)
        raise FileNotFoundError(
            f"Capture path is ambiguous for '{requested}'. Matching files: {choices}"
        )
    available = sorted(
        candidate.name
        for candidate in parent.iterdir()
        if candidate.is_file() and candidate.suffix.casefold() in PCAP_SUFFIXES
    )
    hint = f" Available captures: {', '.join(available)}." if available else ""
    raise FileNotFoundError(f"Capture file does not exist: {requested}.{hint}")


def resolve_capture_paths(requested: Path) -> list[Path]:
    """Resolve one capture or all direct PCAP/PCAPNG files in a protocol folder."""
    if requested.is_file():
        return [resolve_capture_path(requested)]
    if requested.is_dir():
        captures = sorted(
            (
                candidate
                for candidate in requested.iterdir()
                if candidate.is_file() and candidate.suffix.casefold() in PCAP_SUFFIXES
            ),
            key=lambda candidate: candidate.name.casefold(),
        )
        if captures:
            return captures
        raise FileNotFoundError(
            f"No .pcap or .pcapng files were found directly in protocol folder: {requested}"
        )
    return [resolve_capture_path(requested)]
