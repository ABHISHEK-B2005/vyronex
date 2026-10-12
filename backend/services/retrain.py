"""Retraining Service for MongoDB integration (Chunk 8E)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ml.retrain import execute_retraining_pipeline

logger = logging.getLogger(__name__)


async def trigger_model_retrain(
    database: Any,
    exercise: str = "squat",
    force_activate: bool = False,
) -> Dict[str, Any]:
    """Fetch expert-labeled reps from MongoDB and run safe champion-challenger retraining."""
    # Only pull reps that are expert-reviewed
    query = {"expert_reviewed": True}
    
    reps: List[Dict[str, Any]] = []
    if database is not None and hasattr(database, "reps"):
        cursor = database.reps.find(query)
        reps = await cursor.to_list(length=None) if hasattr(cursor, "to_list") else getattr(cursor, "documents", [])

    logger.info("Found %d expert-confirmed reps for retraining", len(reps))

    result = execute_retraining_pipeline(
        expert_reps=reps,
        exercise=exercise,
        force_activate=force_activate,
    )
    return result
