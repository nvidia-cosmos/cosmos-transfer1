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

"""
Monkeypatch Transformers AutoConfig.register to ignore duplicate-key ValueError
caused by vLLM trying to register model types (e.g. "aimv2") that already exist.

IMPORTANT:
 - This module must be imported **before** any import that may load vllm or its
   transformers_utils/configs modules (i.e., at the very top of your entrypoint).
 - It does not import vllm configs (that would trigger the same error).
"""

import warnings

try:
    # import the AutoConfig object to patch
    from transformers.models.auto import configuration_auto as _conf_mod

    AutoConfig = getattr(_conf_mod, "AutoConfig", None)
except Exception as e:
    AutoConfig = None
    warnings.warn(f"[patch] Could not import AutoConfig: {e}")

if AutoConfig is None:
    # Nothing we can do here; warn and exit gracefully.
    warnings.warn("[patch] AutoConfig not available; skipping registry monkeypatch.")
else:
    # save original register
    _orig_register = AutoConfig.register

    def _safe_register(model_type, config, exist_ok: bool = False):
        """
        Wrap the original AutoConfig.register. If a ValueError occurs about
        a duplicate key (e.g. "'aimv2' is already used..."), ignore it.
        For all other errors, re-raise.
        """
        try:
            return _orig_register(model_type, config, exist_ok=exist_ok)
        except ValueError as ve:
            msg = str(ve)
            # Transformers message pattern for duplicate key:
            # "'aimv2' is already used by a Transformers config, pick another name."
            if "already used" in msg and "Transformers config" in msg:
                # ignore duplicate registration attempts
                warnings.warn(f"[patch] Ignoring duplicate AutoConfig registration for '{model_type}'.")
                return None
            # still raise unexpected ValueErrors
            raise

    # apply monkeypatch
    AutoConfig.register = staticmethod(_safe_register)
    warnings.warn("[patch] Patched AutoConfig.register to ignore duplicate-key ValueError.")
