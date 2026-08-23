"""Compatibility exports for validation schemas.

New code should import from :mod:`src.schemas.validation`. Existing imports are
kept working so the schema move is non-breaking.
"""

from src.schemas.validation import (
    ErrorSeverity,
    ErrorType,
    ResponsibleAgent,
    ValidationError,
    ValidationResult,
)

__all__ = [
    "ErrorSeverity",
    "ErrorType",
    "ResponsibleAgent",
    "ValidationError",
    "ValidationResult",
]
