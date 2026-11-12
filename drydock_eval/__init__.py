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
"""drydock-eval: a safety and fidelity kit for evaluating Gemma 4 coding agents locally.

Modules:
  drydock_eval.conform  checks that what a local stack sends to the model equals what the Kaggle scorer sends.
  drydock_eval.contain  reference implementation of a contained runner: one container per job, a request forwarder,
                        a gate file and a canary suite (needs Docker to run; its checks of the request filter do not).
  drydock_eval.compare  decides between agent configurations from a results table: a frozen pass rule, Holm-corrected
                        sign-flip tests and their simulators (needs numpy).
"""

__version__ = "0.4.0"
