"""DuckDB-backed append-only ledger.

One file, columnar, SQL analytics. Every table in the paper becomes one query,
which is what you want the night before a deadline (03_IMPLEMENTATION_PLAN.md 3.6).
JSON-typed fields are stored as VARCHAR holding JSON; DuckDB's json functions
read them back, and this keeps the schema stable as records evolve.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import duckdb
from pydantic import BaseModel

from .schema import TABLES

_SQL_TYPE = {int: "BIGINT", float: "DOUBLE", bool: "BOOLEAN", str: "VARCHAR"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _ddl(name: str, model: type[BaseModel]) -> str:
    cols = []
    for fname, field in model.model_fields.items():
        ann = field.annotation
        t = _SQL_TYPE.get(ann, "VARCHAR")
        cols.append(f'"{fname}" {t}')
    return f'CREATE TABLE IF NOT EXISTS "{name}" ({", ".join(cols)});'


def _row(model: BaseModel) -> list[Any]:
    out = []
    for v in model.model_dump().values():
        out.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
    return out


class Ledger:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        for name, model in TABLES.items():
            self.con.execute(_ddl(name, model))

    def append(self, table: str, records: BaseModel | Iterable[BaseModel]) -> int:
        if isinstance(records, BaseModel):
            records = [records]
        records = list(records)
        if not records:
            return 0
        n = len(TABLES[table].model_fields)
        ph = ", ".join(["?"] * n)
        self.con.executemany(f'INSERT INTO "{table}" VALUES ({ph})', [_row(r) for r in records])
        return len(records)

    def q(self, sql: str, *params):
        return self.con.execute(sql, list(params)).fetchdf()

    def counts(self) -> dict[str, int]:
        return {
            t: int(self.con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0])
            for t in TABLES
        }

    def close(self) -> None:
        self.con.close()
