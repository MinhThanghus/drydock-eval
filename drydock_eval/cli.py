# Copyright 2026 The drydock-eval Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top-level command line: ``drydock-eval <module> <command> ...`` (import name ``drydock_eval``).

Each kit module exposes ``add_subparser(subparsers)`` in ``drydock_eval.<module>.cli``; modules that are not
installed are skipped, so modules can be added independently.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# Known kit modules, in display order. A module missing from the installation is skipped.
MODULES = ("conform", "contain", "compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drydock-eval", description="Gemma 4 agent evaluation kit")
    sub = parser.add_subparsers(dest="module", metavar="<module>")
    sub.required = True
    for name in MODULES:
        try:
            mod = importlib.import_module(f"drydock_eval.{name}.cli")
        except ImportError:
            continue
        mod.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
