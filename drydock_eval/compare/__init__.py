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
"""drydock_eval.compare: decide whether a candidate agent beats a baseline on the same tasks, without fooling yourself.

Three commands over a plain results table (one row per run: arm, task, repository, repeat, solved, and optionally
agent_time and row_class):

``gate``      the frozen five-criterion pass rule (repository-clustered; PASS / FAIL / INCONCLUSIVE with a report)
``signflip``  several arms against one reference: exact two-sided task-level sign-flip test, Holm-corrected, with the
              repository-cluster interval as the secondary analysis
``simulate``  the simulators behind the rule's published size and power numbers

The statistics are vendored, block for block, from the authors' verified analysis scripts (``_vendored_*.py``); the
table loader, the composition and the command line were written for the kit. numpy is required
(``pip install "drydock-eval[compare]"``).
"""

from .table import TableError, load_table  # noqa: F401
