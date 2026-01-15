import os

# Global override for programmatic control
_tracing_enabled_override: bool | None = None


def is_tracing_disabled() -> bool:
    """Check if tracing is disabled via environment variable or programmatic override."""
    if _tracing_enabled_override is not None:
        return not _tracing_enabled_override

    # Check environment variable
    return os.environ.get("CUTEDSL_TRACE_DISABLED", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_tracing_enabled(enabled: bool) -> None:
    """Programmatically enable or disable tracing."""
    global _tracing_enabled_override
    _tracing_enabled_override = enabled
