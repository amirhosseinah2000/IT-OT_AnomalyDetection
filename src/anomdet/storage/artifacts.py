"""Artifact catalog with a filesystem source of truth and optional ClickHouse sink."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pandas.api import types as pd_types

from anomdet.core.io import utc_now, write_json, write_table


class TableMirror(Protocol):
    """Minimal database boundary; the pipeline never depends on a vendor API directly."""

    def write(self, table_name: str, frame: pd.DataFrame) -> None: ...


class NullMirror:
    """No-op default for portable local and CI execution."""

    def write(self, table_name: str, frame: pd.DataFrame) -> None:
        del table_name, frame


class ClickHouseMirror:
    """Lazy ClickHouse writer; credentials are supplied through environment variables."""

    def __init__(self, options: dict[str, Any]) -> None:
        try:
            import clickhouse_connect
        except ImportError as error:  # pragma: no cover - optional integration.
            raise RuntimeError(
                "ClickHouse mirroring is enabled but clickhouse-connect is not installed. "
                "Run `uv sync` after updating dependencies."
            ) from error
        password_env = str(options.get("password_env", "CLICKHOUSE_PASSWORD"))
        self.client = clickhouse_connect.get_client(
            host=str(options.get("host", "localhost")),
            port=int(options.get("port", 8123)),
            username=str(options.get("username", "default")),
            password=os.getenv(password_env),
            database=str(options.get("database", "anomdet")),
            secure=bool(options.get("secure", False)),
        )
        self.database = str(options.get("database", "anomdet"))

    @staticmethod
    def _identifier(value: str) -> str:
        if not value.replace("_", "").isalnum():
            raise ValueError(f"Unsafe ClickHouse identifier: {value!r}")
        return value

    @staticmethod
    def _column_type(dtype: Any) -> str:
        if pd_types.is_bool_dtype(dtype):
            return "Nullable(UInt8)"
        if pd_types.is_integer_dtype(dtype):
            return "Nullable(Int64)"
        if pd_types.is_float_dtype(dtype):
            return "Nullable(Float64)"
        if pd_types.is_datetime64_any_dtype(dtype):
            return "Nullable(DateTime64(6, 'UTC'))"
        return "Nullable(String)"

    def _ensure_table(self, table_name: str, frame: pd.DataFrame) -> None:
        table = self._identifier(table_name)
        columns = ", ".join(
            f"`{self._identifier(str(column))}` {self._column_type(dtype)}"
            for column, dtype in frame.dtypes.items()
        )
        self.client.command(f"CREATE DATABASE IF NOT EXISTS `{self._identifier(self.database)}`")
        self.client.command(
            f"CREATE TABLE IF NOT EXISTS `{self._identifier(self.database)}`.`{table}` "
            f"({columns}) ENGINE = MergeTree ORDER BY tuple()"
        )

    def write(self, table_name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self._ensure_table(table_name, frame)
        payload = frame.copy()
        for column in payload.select_dtypes(include=["bool"]).columns:
            payload[column] = payload[column].astype("UInt8")
        for column in payload.select_dtypes(
            exclude=["number", "bool", "datetime", "datetimetz"]
        ).columns:
            payload[column] = payload[column].astype("string").where(payload[column].notna(), None)
        self.client.insert_df(table_name, payload, database=self.database)


def _schema_hash(frame: pd.DataFrame) -> str:
    schema = [(column, str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass
class ArtifactStore:
    """Write self-describing assets and one queryable catalog per run."""

    root: Path
    mirror: TableMirror = field(default_factory=NullMirror)
    catalog: list[dict[str, Any]] = field(default_factory=list)

    def table(
        self,
        relative_path: str | Path,
        frame: pd.DataFrame,
        *,
        kind: str,
        protocol: str | None = None,
        split: str | None = None,
        description: str = "",
        mirror_table: str | None = None,
    ) -> Path:
        path = self.root / relative_path
        write_table(frame, path)
        self.catalog.append(
            {
                "artifact_id": hashlib.sha256(
                    str(path.relative_to(self.root)).encode()
                ).hexdigest()[:16],
                "kind": kind,
                "protocol": protocol,
                "split": split,
                "format": path.suffix.lstrip("."),
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "schema_hash": _schema_hash(frame),
                "description": description,
                "created_at": utc_now(),
            }
        )
        if mirror_table:
            self.mirror.write(mirror_table, frame)
        return path

    def register_existing(
        self,
        path: Path,
        *,
        kind: str,
        protocol: str | None = None,
        split: str | None = None,
        description: str = "",
        rows: int | None = None,
        columns: int | None = None,
    ) -> None:
        """Catalog a file written by an external producer without copying it again."""
        relative = path.relative_to(self.root)
        self.catalog.append(
            {
                "artifact_id": hashlib.sha256(str(relative).encode()).hexdigest()[:16],
                "kind": kind,
                "protocol": protocol,
                "split": split,
                "format": path.suffix.lstrip("."),
                "path": str(relative).replace("\\", "/"),
                "rows": rows,
                "columns": columns,
                "schema_hash": None,
                "description": description,
                "created_at": utc_now(),
            }
        )

    def json(
        self,
        relative_path: str | Path,
        payload: dict[str, Any],
        *,
        kind: str,
        description: str = "",
    ) -> Path:
        path = self.root / relative_path
        write_json(payload, path)
        self.catalog.append(
            {
                "artifact_id": hashlib.sha256(
                    str(path.relative_to(self.root)).encode()
                ).hexdigest()[:16],
                "kind": kind,
                "protocol": payload.get("protocol"),
                "split": payload.get("split"),
                "format": "json",
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "rows": None,
                "columns": None,
                "schema_hash": None,
                "description": description,
                "created_at": utc_now(),
            }
        )
        return path

    def finalize(self) -> Path:
        """Persist catalog as Parquet for programs and JSON for quick manual review."""
        frame = pd.DataFrame(self.catalog)
        path = self.root / "catalog" / "artifacts.parquet"
        write_table(frame, path)
        write_json(
            {"created_at": utc_now(), "artifact_count": len(frame), "catalog": str(path)},
            self.root / "catalog" / "summary.json",
        )
        return path


def create_artifact_store(config: dict[str, Any], root: Path) -> ArtifactStore:
    """Create a local store and optionally attach an independently replaceable mirror."""
    settings = config.get("storage", {}).get("clickhouse", {})
    mirror: TableMirror = (
        ClickHouseMirror(settings) if settings.get("enabled", False) else NullMirror()
    )
    return ArtifactStore(root=root, mirror=mirror)
