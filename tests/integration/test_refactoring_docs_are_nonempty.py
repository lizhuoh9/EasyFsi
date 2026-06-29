from __future__ import annotations

import unittest
from pathlib import Path


REFACTORING_DOCS = Path("docs") / "refactoring"


class RefactoringDocsNonEmptyTests(unittest.TestCase):
    def test_refactoring_docs_are_nonempty_or_reserved_with_reason(self):
        empty_or_invalid_reserved: list[str] = []

        for path in sorted(REFACTORING_DOCS.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                empty_or_invalid_reserved.append(path.as_posix())
                continue
            if text.startswith("# Reserved"):
                if "Reason:" not in text:
                    empty_or_invalid_reserved.append(path.as_posix())

        self.assertEqual(empty_or_invalid_reserved, [])


if __name__ == "__main__":
    unittest.main()
