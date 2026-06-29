from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.validation.ansys_vertical_flap import fluent_reference_collection as _impl  # noqa: E402
from tools.validation.ansys_vertical_flap.fluent_reference_collection import *  # noqa: F401,F403,E402


def __getattr__(name: str):
    return getattr(_impl, name)


def main() -> int:
    try:
        payload = _impl.run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[fluent_reference_collection] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[fluent_reference_collection] wrote "
        f"{payload['candidate_status']} to {_impl.OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
