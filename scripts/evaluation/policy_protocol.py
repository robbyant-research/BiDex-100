"""NumPy-aware msgpack helpers shared by policy servers and clients."""

from __future__ import annotations

import msgpack
import numpy as np


_MAX_CONTAINER_SIZE = 100_000_000


def packb(value) -> bytes:
    """Serialize a msgpack value, encoding NumPy arrays without copying shape data."""

    def encode(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__nd__": True,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "data": obj.tobytes(),
            }
        raise TypeError(f"unhandled {type(obj)}")

    return msgpack.packb(value, default=encode, use_bin_type=True)


def unpackb(payload: bytes):
    """Deserialize values produced by :func:`packb`."""

    def decode(obj):
        if isinstance(obj, dict) and obj.get("__nd__"):
            return np.frombuffer(
                obj["data"], dtype=np.dtype(obj["dtype"])
            ).reshape(obj["shape"])
        return obj

    return msgpack.unpackb(
        payload,
        object_hook=decode,
        raw=False,
        max_bin_len=_MAX_CONTAINER_SIZE,
        max_str_len=_MAX_CONTAINER_SIZE,
        max_array_len=_MAX_CONTAINER_SIZE,
        max_map_len=_MAX_CONTAINER_SIZE,
    )
