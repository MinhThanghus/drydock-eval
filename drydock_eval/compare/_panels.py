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
"""Repository index of every task of the two paper panels, in panel order (derived data).

Generated from the authors' frozen panel files: for each panel, the repository names were sorted and every
task was replaced by the position of its repository in that list. No task identifier or repository name is
stored. With these vectors ``simulate`` draws the same panels as the authors' simulation runs.
"""

REPO_INDEX = {
    "small": [  # 72 tasks, 18 repositories
        7, 11, 8, 7, 11, 13, 11, 7, 17, 7, 4, 17, 11, 13, 11, 8, 17, 10, 11, 14, 17, 0, 9, 7, 3, 8, 2, 10, 7, 11,
        11, 12, 8, 16, 3, 3, 8, 11, 9, 4, 5, 5, 11, 11, 4, 11, 5, 4, 17, 7, 3, 10, 16, 8, 0, 17, 15, 10, 2, 3, 5,
        4, 17, 16, 12, 6, 13, 3, 16, 8, 10, 1,
    ],
    "large": [  # 200 tasks, 20 repositories
        8, 12, 9, 8, 12, 14, 12, 8, 19, 8, 5, 19, 12, 14, 12, 9, 19, 11, 12, 15, 19, 1, 10, 8, 4, 9, 3, 11, 8, 12,
        12, 13, 9, 17, 4, 4, 9, 12, 10, 5, 6, 6, 12, 12, 5, 12, 6, 5, 19, 8, 4, 11, 17, 9, 1, 12, 19, 16, 11, 3, 4,
        6, 5, 19, 17, 13, 7, 14, 4, 17, 12, 9, 12, 11, 2, 12, 10, 11, 9, 9, 8, 11, 17, 19, 10, 9, 9, 3, 9, 6, 11,
        16, 9, 11, 2, 14, 11, 11, 19, 11, 9, 8, 11, 11, 11, 9, 11, 6, 6, 6, 14, 8, 17, 2, 4, 19, 10, 8, 5, 8, 3, 8,
        3, 1, 14, 6, 19, 5, 9, 5, 19, 18, 7, 10, 14, 10, 19, 19, 5, 6, 4, 6, 8, 16, 19, 5, 6, 19, 10, 3, 10, 5, 8,
        5, 8, 4, 14, 16, 2, 14, 10, 15, 7, 3, 4, 17, 1, 6, 10, 7, 0, 6, 17, 10, 16, 5, 10, 6, 7, 6, 10, 4, 0, 5,
        16, 17, 15, 17, 7, 10, 16, 10, 3, 15, 3, 2, 16, 17, 14, 16,
    ],
}
