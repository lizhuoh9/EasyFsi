"""Bounded atomic publication when Windows readers temporarily deny rename."""

from __future__ import annotations

import errno
import os
import time


_WINDOWS_REPLACE_RETRY_DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.32)
_WINDOWS_REPLACE_CONFLICTS = frozenset((5, 32, 33))


def replace_file_atomically(
    source: str | os.PathLike[str], destination: str | os.PathLike[str],
) -> None:
    """Replace once prepared bytes, never delete the old destination first.

    Native Windows readers can deny rename briefly; WSL UNC reports this as
    WinError 5 rather than 32. Retry only these Windows error codes, for at most
    eight attempts and 0.95 seconds of backoff. A permanent denial still raises
    the last original error. Other I/O errors escape immediately.

    This does not retry serialization, physical steps, or accepted-state work.
    The caller remains responsible for fsync and cleaning its owned temp file.
    """
    for delay_s in (*_WINDOWS_REPLACE_RETRY_DELAYS_S, None):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in _WINDOWS_REPLACE_CONFLICTS or delay_s is None:
                raise
            time.sleep(delay_s)


def publish_file_create_only(
    source: str | os.PathLike[str], destination: str | os.PathLike[str],
) -> None:
    """Publish once-prepared bytes without replacing an existing destination.

    Windows can transiently deny even a create-only rename while a reader or
    scanner holds the directory. Retry only the same bounded sharing-conflict
    codes used by atomic replacement. An existing destination is never retried
    or overwritten; it is reported immediately for caller-side semantic checks.
    """

    for delay_s in (*_WINDOWS_REPLACE_RETRY_DELAYS_S, None):
        try:
            if os.name == "nt":
                os.rename(source, destination)
            else:
                os.link(source, destination)
            return
        except OSError as error:
            if error.errno == errno.EEXIST or getattr(error, "winerror", None) == 183:
                raise FileExistsError(destination) from error
            retryable = (
                os.name == "nt"
                and getattr(error, "winerror", None) in _WINDOWS_REPLACE_CONFLICTS
                and delay_s is not None
            )
            if not retryable:
                raise
            time.sleep(delay_s)
