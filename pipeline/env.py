"""Runtime environment checks (XGBoost's OpenMP dependency on macOS)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FLAG = "XPTS_LIBOMP_REEXEC"


def ensure_xgboost_importable() -> None:
    """Import xgboost; on macOS without libomp, re-exec once using scikit-learn's bundled libomp.

    The proper fix is `brew install libomp`. The fallback only adds scikit-learn's own copy of the
    OpenMP runtime to the dynamic-library search path for this process.
    """
    try:
        import xgboost  # noqa: F401
        return
    except Exception as exc:  # XGBoostError subclasses Exception
        if sys.platform != "darwin" or os.environ.get(_FLAG):
            raise RuntimeError(f"xgboost cannot be imported ({exc}). On macOS run `brew install libomp`.") from exc
        import sklearn
        dylibs = Path(sklearn.__file__).parent / ".dylibs"
        if not (dylibs / "libomp.dylib").exists():
            raise RuntimeError("xgboost needs libomp: run `brew install libomp`") from exc
        env = dict(os.environ, **{_FLAG: "1"})
        env["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(dylibs), env.get("DYLD_FALLBACK_LIBRARY_PATH")]))
        print(f"[xpts] libomp not found; re-running with scikit-learn's bundled copy ({dylibs}). "
              "Install it permanently with `brew install libomp`.", file=sys.stderr)
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
