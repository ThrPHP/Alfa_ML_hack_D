"""Append-only experiment ledger.

No mutable "best result" pointer file is kept on purpose: a pointer file
that gets overwritten is exactly the kind of thing a killed-mid-write
process (3-hour hard timeout!) could corrupt or leave stale. Instead every
run_experiment() call appends one line to a .jsonl file, and select_best()
recomputes the best entry from that log on demand. Reading back a full
history across a fresh subprocess (the "every run_python call is a new
process" pattern from the organizers' baseline.py) only costs an append-only
file read.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.contracts import ExperimentResult
from ml_core.io_utils import append_jsonl, read_jsonl

LOG_FILENAME = "experiments.jsonl"


def log_path(experiments_dir: str | Path) -> Path:
    return Path(experiments_dir) / LOG_FILENAME


def record(result: ExperimentResult, experiments_dir: str | Path) -> None:
    append_jsonl(result.model_dump(mode="json"), log_path(experiments_dir))


def load_all(experiments_dir: str | Path) -> list[ExperimentResult]:
    return [ExperimentResult.model_validate(r) for r in read_jsonl(log_path(experiments_dir))]


def select_best(
    experiments_dir: str | Path,
    validation_id: str | None = None,
    require_fully_comparable: bool = True,
) -> ExperimentResult | None:
    """Best ExperimentResult so far, or None if nothing qualifies.

    By default only considers results that ran every requested fold (so a
    run truncated by a time budget never silently wins against a complete
    one) and, when validation_id is given, only results scored on that exact
    split — comparing across different validation plans is apples-to-oranges.
    """
    results = load_all(experiments_dir)
    candidates = [r for r in results if r.status == "success" and r.metric_value is not None]
    if validation_id is not None:
        candidates = [r for r in candidates if r.validation_id == validation_id]
    if require_fully_comparable:
        candidates = [r for r in candidates if r.is_fully_comparable]
    if not candidates:
        return None
    first = candidates[0]
    return max(
        candidates,
        key=lambda r: r.metric_value if first.greater_is_better else -r.metric_value,
    )
