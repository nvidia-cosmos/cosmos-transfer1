# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

import termcolor

parser = argparse.ArgumentParser(description="Cosmos IP header checker/fixer")
parser.add_argument("--fix", action="store_true", help="apply the fixes instead of checking")
args, files_to_check = parser.parse_known_args()


def get_header(ext: str = "py", old: str | bool = False) -> list[str]:
    # This is the raw header.
    # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    header = [
        "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        'Licensed under the Apache License, Version 2.0 (the "License");',
        "you may not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "http://www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        'distributed under the License is distributed on an "AS IS" BASIS,',
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
    ]
    # Reformat according to different file extensions.
    if ext == ".py" and old:
        if old == "single":
            header = ["'''"] + header + ["'''"]
        elif old == "double":
            header = ['"""'] + header + ['"""']
        else:
            raise NotImplementedError
    elif ext in (".py", ".yaml"):
        header = [("# " + line if line else "#") for line in header]
    elif ext in (".c", ".cpp", ".cu", ".h", ".cuh"):
        header = ["/*"] + [(" * " + line if line else " *") for line in header] + [" */"]
    else:
        raise NotImplementedError
    return header


def get_header_ea(ext: str = "py", old: str | bool = False) -> list[str]:
    # This is the raw header.
    # The early-access software is governed by the NVIDIA Evaluation License Agreement – EA Cosmos Code (v. Feb 2025).
    # The license reference will be the finalized version of the license linked above.
    header = [
        "The early-access software is governed by the NVIDIA Evaluation License Agreement – EA Cosmos Code (v. Feb 2025).",
        "The license reference will be the finalized version of the license linked above.",
    ]
    # Reformat according to different file extensions.
    if ext == ".py" and old:
        if old == "single":
            header = ["'''"] + header + ["'''"]
        elif old == "double":
            header = ['"""'] + header + ['"""']
        else:
            raise NotImplementedError
    elif ext in (".py", ".yaml"):
        header = [("# " + line if line else "#") for line in header]
    elif ext in (".c", ".cpp", ".cu", ".h", ".cuh"):
        header = ["/*"] + [(" * " + line if line else " *") for line in header] + [" */"]
    else:
        raise NotImplementedError
    return header


def apply_file(file: str, results: dict[str, int], fix: bool = False) -> None:
    if file.endswith("__init__.py"):
        return
    ext = os.path.splitext(file)[1]
    # Read the file content (line by line).
    content = open(file).read().splitlines()
    # Check if cosmos_transfer1 header (with a blank newline) is properly embedded.
    header = get_header(ext=ext)
    header_ea = get_header_ea(ext=ext)
    if fix:
        # If header passes format check, then just exit
        if _check_header(content, header):
            return
        if _check_header(content, header_ea):
            return
        print(f"fixing: {file}")
        # Clean up leading blank lines.
        while len(content) > 0 and not content[0]:
            content.pop(0)
        # Add cosmos_trasnfer1 copyright header.
        content = header + [""] + content
        # Write content back to file.
        with open(file, "w") as file_obj:
            for line in content:
                file_obj.write(line + "\n")
    else:
        if not _check_header(content, header) and not _check_header(content, header_ea):
            bad_header = colorize("BAD HEADER", color="red", bold=True)
            print(f"{bad_header}: {file}")
            results[file] = 1
        else:
            results[file] = 0


def traverse_directory(path: str, results: dict[str, int], fix: bool = False, substrings_to_skip=[]) -> None:
    # Apply/check the header for an entire directory.
    files = os.listdir(path)
    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            # Traverse into the subdirectory.
            traverse_directory(full_path, results, fix=fix, substrings_to_skip=substrings_to_skip)
        elif os.path.isfile(full_path):
            # Process the file.
            ext = os.path.splitext(file)[1]
            to_skip = False
            for substr in substrings_to_skip:
                if substr in full_path:
                    to_skip = True
                    break

            if not to_skip and ext in (".py", ".yaml", ".c", ".cpp", ".cu", ".h", ".cuh"):
                apply_file(full_path, results, fix=fix)
        else:
            raise NotImplementedError


def _check_header(content: list[str], header: list[str]) -> bool:
    if content[: len(header)] != header:
        return False
    if len(content) > len(header):
        if len(content) == len(header) + 1:
            return False
        if not (content[len(header)] == "" and content[len(header) + 1] != ""):
            return False
    return True


def colorize(x: str, color: str, bold: bool = False) -> str:
    return termcolor.colored(str(x), color=color, attrs=("bold",) if bold else None)  # type: ignore


if __name__ == "__main__":
    if not files_to_check:
        # Default to the entire Cosmos_trasnfer1 repo.
        files_to_check = [
            "cosmos_transfer1/auxiliary",
            "cosmos_transfer1/diffusion",
            "cosmos_transfer1/utils",
        ]

    # Check whether all input files/directories are valid.
    for file in files_to_check:
        assert os.path.isfile(file) or os.path.isdir(file), f"{file} is neither a directory or a file!"

    substrings_to_skip = ["prompt_upsampler"]
    # Run the program.
    results = dict()
    for file in files_to_check:
        if os.path.isfile(file):
            apply_file(file, results, fix=args.fix)
        elif os.path.isdir(file):
            traverse_directory(file, results, fix=args.fix, substrings_to_skip=["prompt_upsampler"])
        else:
            raise NotImplementedError

    if any(results.values()):
        sys.exit(1)
