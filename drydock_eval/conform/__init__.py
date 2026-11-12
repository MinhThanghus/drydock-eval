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
"""drydock_eval.conform: check that what a local stack sends to the model equals what the Kaggle scorer sends.

Pieces: a recording proxy and a scripted mock server that follows vLLM 0.19.1's message parse path (``recorder``,
``mock``), eight checks over captured requests plus a reference built from a capture of your own harness
(``checks``), the vLLM 0.19.1 request-field table (``vllm_wire``), live google-adk demos (``demo``), and a mutation
suite that measures how well the checks detect each kind of drift (``mutate``). ``graphcheck`` checks a task's
code-graph JSON and embedding NPZ the way the harness's graph tools use them (``drydock-eval conform graph``).
"""
