from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence

from cases import AVAILABLE_CASES, CASE_MODULES


def _load_case_main(case_name: str) -> Callable[[list[str] | None], dict[str, object]]:
    module_name = CASE_MODULES[case_name]
    module = importlib.import_module(module_name)
    case_main = getattr(module, "main", None)
    if not callable(case_main):
        available = "\n".join(f"  {case}" for case in AVAILABLE_CASES)
        raise SystemExit(
            f"case '{case_name}' has no main() entry point and cannot be run "
            f"from this dispatcher.\n\nAvailable cases:\n{available}"
        )
    return case_main


def _usage() -> str:
    cases = "\n".join(f"  {case}" for case in AVAILABLE_CASES)
    return f"Usage: python run_simulation.py <case> [case args]\n\nAvailable cases:\n{cases}"


def dispatch(argv: Sequence[str]) -> dict[str, object]:
    args = list(argv)
    if not args:
        raise SystemExit(_usage())
    if args[0] in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)
    case_name = args[0]
    if case_name not in CASE_MODULES:
        raise SystemExit(f"Unknown case '{args[0]}'.\n\n{_usage()}")
    return _load_case_main(case_name)(args[1:])


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return dispatch(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
