# Unsloth must be imported before transformers to apply its optimizations
import os as _os, sys as _sys

if _os.environ.get("NO_UNSLOTH", "").strip().lower() not in ("1", "true", "yes") \
        and not any(a in _sys.argv for a in ("--no_unsloth", "--no-unsloth")):
    try:
        import unsloth  # noqa: F401
    except Exception:
        pass
