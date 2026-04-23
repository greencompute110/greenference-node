"""Auth validation for unified node agent endpoints."""

from __future__ import annotations

from fastapi import HTTPException, status


def validate_optional_auth(header: str | None, expected: str | None) -> None:
    """Validate X-Agent-Auth or X-Compute-Auth header if a secret is configured."""
    if expected is None:
        return  # Auth not configured, allow all
    if not header or header != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing auth header",
        )
