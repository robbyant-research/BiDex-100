"""Per-task init-state library.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INIT_STATES_DIR = os.path.join(_PKG_ROOT, "tasks_data", "init_states")


def init_states_path(suite: str, name: str, init_states_file: Optional[str] = None) -> str:
    """Return the absolute path for a task's init-states .npy.

    If `init_states_file` is given (as in Task.init_states_file), it wins;
    relative paths are resolved against ``INIT_STATES_DIR/<suite>/``. Otherwise
    we default to ``<suite>/<name>.npy``.
    """
    if init_states_file:
        if os.path.isabs(init_states_file):
            return init_states_file
        return os.path.join(INIT_STATES_DIR, suite, init_states_file)
    return os.path.join(INIT_STATES_DIR, suite, f"{name}.npy")


def load_init_states(path: str) -> np.ndarray:
    """Return a (N, 1+nq+nv) float64 array. Missing file -> empty array of shape (0, 0)."""
    if not os.path.isfile(path):
        return np.zeros((0, 0), dtype=np.float64)
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]   # one snapshot saved with --save-init-state ends up 1-D
    return arr


def save_init_states(path: str, states: np.ndarray) -> None:
    """Save a (N, 1+nq+nv) array, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(states))
