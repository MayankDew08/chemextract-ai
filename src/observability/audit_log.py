"""Append-only JSONL audit log for pipeline runs.

This file contains log_event and small domain helpers. Every extraction run
appends one JSON object per line to a single shared file so failures can be
audited later without querying SQLite or replaying terminal logs. Audit
failures must never break extraction, so every write is best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def audit_path() -> Path:
    """Return the configured audit file path, defaulting to data/audit.jsonl."""

    return Path(os.getenv("AUDIT_LOG_PATH", "./data/audit.jsonl"))


def log_event(event: str, **fields: Any) -> None:
    """Append one JSON line to the audit file without ever raising."""

    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        record.update(fields)
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _write_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning("Audit log write failed for event '%s': %s", event, exc)


def log_recipe_result(recipe: Any) -> None:
    """Record the final outcome of one recipe including corrections and methods."""

    try:
        log_event(
            "recipe_result",
            recipe_id=recipe.recipe_id,
            status=recipe.validation_status.value,
            llm_provider=recipe.llm_provider,
            llm_model=recipe.llm_model,
            extraction_methods=recipe.node_extraction_methods,
            tokens=recipe.total_tokens_used,
            cost_usd=recipe.estimated_cost_usd,
            latency_seconds=recipe.total_latency_seconds,
            corrections=[
                {
                    "attempt_number": record.attempt_number,
                    "error_type": record.error_type,
                    "error_field": record.error_field,
                    "error_message": record.error_message,
                    "agent_routed_to": record.agent_routed_to,
                }
                for record in recipe.correction_history
            ],
            validation_errors=recipe.validation_errors,
            warnings=[warning.error_type.value for warning in recipe.validation_warnings],
        )
    except Exception as exc:
        logger.warning("Audit log could not serialize recipe result: %s", exc)


def log_parse_failure(agent: str, raw_output: str, detail: str = "") -> None:
    """Record the verbatim LLM output that failed structured parsing."""

    log_event(
        "parse_failure",
        agent=agent,
        detail=detail,
        raw_output=raw_output,
    )
