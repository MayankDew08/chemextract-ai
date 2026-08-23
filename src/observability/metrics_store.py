"""SQLite-backed metrics persistence for Phase 5 observability.

This file contains MetricsStore. Every completed recipe run is recorded in a
small SQLite table so dashboards can compute durable metrics across API calls.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from src.observability.metrics_models import ErrorFrequency, NodeTiming, PipelineMetrics
from src.schemas.recipe import ChemicalRecipe
from src.storage.base import BaseGraphStore, NodeType

logger = logging.getLogger(__name__)


class MetricsStore:
    """MetricsStore records completed recipe runs and computes dashboard snapshots."""

    def __init__(self, db_path: str = "./data/metrics.db") -> None:
        """Store the SQLite path so initialization can happen in FastAPI lifespan."""

        self._db_path = db_path
        self._conn = None

    async def initialize(self) -> None:
        """Open SQLite and create metrics tables idempotently."""

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create the recipe_runs table if it does not exist."""

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_runs (
                recipe_id         TEXT PRIMARY KEY,
                validation_status TEXT NOT NULL,
                correction_count  INTEGER DEFAULT 0,
                total_tokens      INTEGER DEFAULT 0,
                cost_usd          REAL DEFAULT 0.0,
                latency_seconds   REAL DEFAULT 0.0,
                llm_model         TEXT DEFAULT '',
                error_types       TEXT DEFAULT '[]',
                warning_types     TEXT DEFAULT '[]',
                node_latencies    TEXT DEFAULT '{}',
                node_tokens       TEXT DEFAULT '{}',
                extracted_at      TEXT
            )
            """
        )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(recipe_runs)")}
        if "warning_types" not in columns:
            self._conn.execute("ALTER TABLE recipe_runs ADD COLUMN warning_types TEXT DEFAULT '[]'")
        self._conn.commit()

    async def record_recipe(self, recipe: ChemicalRecipe) -> None:
        """Record one completed recipe run with idempotent upsert semantics."""

        error_types = json.dumps([record.error_type for record in recipe.correction_history])
        warning_types = json.dumps([warning.error_type.value for warning in recipe.validation_warnings])
        self._conn.execute(
            """
            INSERT OR REPLACE INTO recipe_runs (
                recipe_id,
                validation_status,
                correction_count,
                total_tokens,
                cost_usd,
                latency_seconds,
                llm_model,
                error_types,
                warning_types,
                node_latencies,
                node_tokens,
                extracted_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                recipe.recipe_id,
                recipe.validation_status.value,
                recipe.correction_count(),
                recipe.total_tokens_used,
                recipe.estimated_cost_usd,
                recipe.total_latency_seconds,
                recipe.llm_model,
                error_types,
                warning_types,
                json.dumps(recipe.node_latencies),
                json.dumps(recipe.node_tokens),
                recipe.extracted_at.isoformat(),
            ),
        )
        self._conn.commit()

    async def clear(self) -> None:
        """Delete all recorded recipe runs while keeping the database open."""

        if self._conn is None:
            raise RuntimeError("MetricsStore is not initialized")
        self._conn.execute("DELETE FROM recipe_runs")
        self._conn.commit()

    async def compute_metrics(self, graph_store: BaseGraphStore) -> PipelineMetrics:
        """Compute the full dashboard snapshot from stored recipe runs."""

        cursor = self._conn.execute(
            """
            SELECT
                recipe_id,
                validation_status,
                correction_count,
                total_tokens,
                cost_usd,
                latency_seconds,
                llm_model,
                error_types,
                node_latencies,
                node_tokens,
                extracted_at
            FROM recipe_runs
            """
        )
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        if not rows:
            return PipelineMetrics(
                total_recipes=0,
                total_chunks_processed=0,
                passed_first_try=0,
                passed_after_correction=0,
                failed_unresolvable=0,
                success_rate_percent=0.0,
                self_correction_rate_percent=0.0,
                total_tokens_used=0,
                total_cost_usd=0.0,
                avg_latency_seconds=0.0,
                avg_tokens_per_recipe=0.0,
                error_frequencies=[],
                most_common_error=None,
                total_corrections_made=0,
                avg_corrections_per_recipe=0.0,
                node_timings=[],
                graph_node_count=graph_store.node_count(),
                graph_edge_count=graph_store.edge_count(),
                unique_chemicals=_unique_chemical_count(graph_store),
            )
        total = len(rows)
        passed_first = sum(1 for row in rows if row[1] == "PASSED" and row[2] == 0)
        corrected = sum(1 for row in rows if row[1] == "CORRECTED")
        failed = sum(1 for row in rows if row[1] == "FAILED")
        total_tokens = sum(row[3] for row in rows)
        total_cost = sum(row[4] for row in rows)
        total_latency = sum(row[5] for row in rows)
        total_corrections = sum(row[2] for row in rows)
        all_errors = []
        for row in rows:
            all_errors.extend(json.loads(row[7] or "[]"))
        error_counts: dict[str, int] = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        error_frequencies = sorted(
            [
                ErrorFrequency(
                    error_type=error_type,
                    count=count,
                    percentage=round(count / max(len(all_errors), 1) * 100, 1),
                )
                for error_type, count in error_counts.items()
            ],
            key=lambda item: item.count,
            reverse=True,
        )
        node_latency_totals: dict[str, list[float]] = {}
        node_token_totals: dict[str, list[int]] = {}
        for row in rows:
            for node, latency in json.loads(row[8] or "{}").items():
                node_latency_totals.setdefault(node, []).append(float(latency))
            for node, tokens in json.loads(row[9] or "{}").items():
                node_token_totals.setdefault(node, []).append(int(tokens))
        node_timings = [
            NodeTiming(
                node_name=node,
                avg_latency_seconds=round(sum(latencies) / len(latencies), 3),
                total_calls=len(latencies),
                total_tokens=sum(node_token_totals.get(node, [])),
            )
            for node, latencies in node_latency_totals.items()
        ]
        return PipelineMetrics(
            total_recipes=total,
            total_chunks_processed=total,
            passed_first_try=passed_first,
            passed_after_correction=corrected,
            failed_unresolvable=failed,
            success_rate_percent=round((passed_first + corrected) / total * 100, 1),
            self_correction_rate_percent=round(corrected / total * 100, 1),
            total_tokens_used=total_tokens,
            total_cost_usd=round(total_cost, 6),
            avg_latency_seconds=round(total_latency / total, 2),
            avg_tokens_per_recipe=round(total_tokens / total, 0),
            error_frequencies=error_frequencies,
            most_common_error=error_frequencies[0].error_type if error_frequencies else None,
            total_corrections_made=total_corrections,
            avg_corrections_per_recipe=round(total_corrections / total, 2),
            node_timings=node_timings,
            graph_node_count=graph_store.node_count(),
            graph_edge_count=graph_store.edge_count(),
            unique_chemicals=_unique_chemical_count(graph_store),
        )

    async def close(self) -> None:
        """Close the SQLite connection if it was opened."""

        if self._conn:
            self._conn.close()


def _unique_chemical_count(graph_store: BaseGraphStore) -> int:
    """Count chemical nodes from graph export without backend-specific access."""

    graph = graph_store.export_graph_json()
    return sum(1 for node in graph.nodes if node.node_type == NodeType.CHEMICAL)
