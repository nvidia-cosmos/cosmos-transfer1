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

from __future__ import annotations

import attrs

from typing import TypeVar

T = TypeVar("T")

def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls: T) -> T:
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:  # noqa: ANN001
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)  # type: ignore

    cls.__setattr__ = setattr_override  # type: ignore

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True  # type: ignore

    cls.freeze = freeze  # type: ignore

    return cls

@make_freezable
@attrs.define(slots=False)
class DDPConfig:
    # Traverse the computation graph to find parameters that don't receive gradients.
    find_unused_parameters: bool = False
    # Set to True if the computation graph does not change during the whole training loop.
    static_graph: bool = True
    # Set to True if we want to synchronize buffers. Set to False if the sync is going to be handled elsewhere.
    broadcast_buffers: bool = True 