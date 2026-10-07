#!/usr/bin/env python3
"""Obtain and verify the one source file this repository does not redistribute.

FActScore's human-annotation release is the only third-party data the pipeline
needs as a local file: it is not a Hugging Face dataset, so there is no revision
to pin and no loader to point at it. It is not shipped here — every other source
is fetched by the reader from its own publisher, and this one is no different.

Why the checksum matters more here than anywhere else. The FActScore builder keys
each instance by its LINE NUMBER in this file. Its siblings ChatGPT.jsonl and
PerplexityAI.jsonl share the schema exactly, so pointing at one of those, or at a
re-download in a different order, produces 330 perfectly plausible instances
joined to entirely different biographies — with nothing raising. The four Hub
datasets are protected from that by a pinned revision; this file is protected by
FACTSCORE_SHA256, and by nothing else.

Usage:
    ./tools/fetch_factscore.py            # check for the file and verify it
    ./tools/fetch_factscore.py --path X   # verify a copy sitting elsewhere
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root; this file lives in tools/
from raim.tasks import FACTSCORE_SHA256  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DEFAULT = HERE / "datasets" / "InstructGPT.jsonl"
SOURCE = "https://github.com/shmsw25/FActScore"
EXPECTED_LINES = 183

INSTRUCTIONS = f"""
  How to obtain it
  ----------------
  1. Open {SOURCE}
  2. Follow the Google Drive link in their README (the human-annotation data
     from Section 3 of the paper).
  3. Take  data/labeled/InstructGPT.jsonl  -- NOT ChatGPT.jsonl and NOT
     PerplexityAI.jsonl, which share its schema and would be accepted by every
     check except the checksum below.
  4. Place it at  datasets/InstructGPT.jsonl  (or set FACTSCORE_PATH).
  5. Re-run this script to verify it.

  Cite: Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer and Hajishirzi.
  "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form
  Text Generation." EMNLP 2023.  The file is MIT licensed; see NOTICE.
"""


def verify(path: Path) -> int:
    if not path.exists():
        print(f"NOT FOUND: {path}")
        print(INSTRUCTIONS)
        return 1

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = sum(1 for line in path.open() if line.strip())

    print(f"  path   {path}")
    print(f"  lines  {lines} (expected {EXPECTED_LINES})")
    print(f"  sha256 {digest}")

    if digest == FACTSCORE_SHA256:
        print("\nOK: this is the file the reported verdicts were produced against.")
        return 0

    print(f"\nMISMATCH — expected sha256 {FACTSCORE_SHA256}")
    if lines != EXPECTED_LINES:
        print(f"  The line count differs too ({lines} vs {EXPECTED_LINES}), which "
              f"suggests a different file altogether rather than a re-encoding.")
    else:
        print("  The line count matches, so this may be a sibling release "
              "(ChatGPT.jsonl / PerplexityAI.jsonl) or a differently ordered "
              "re-download. Either would silently re-point every instance.")
    print(INSTRUCTIONS)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=Path, default=DEFAULT,
                    help=f"where the file is (default: {DEFAULT.relative_to(HERE)})")
    args = ap.parse_args()
    sys.exit(verify(args.path))


if __name__ == "__main__":
    main()
