#!/usr/bin/env python3
"""Warm the openpi policy server's JAX JIT compile BEFORE eval workers connect.

Problem: 7 vectorized workers × N envs storm the server with first-infers at once.
The server's first infer triggers JAX sample_actions JIT compile (~1-2 min) which
BLOCKS its asyncio loop -> websocket pings not serviced -> client connections
timeout (~40s) and close -> workers' client.infer hangs forever on recv (deadlock).

Fix: connect with ping_interval=None (no ping timeout -> survives the long compile),
send one dummy infer to trigger the compile, wait for the response. After this the
server is warm (compiled) and workers' infers are fast (<1s) -> no timeout.

Usage: python warm_openpi_server.py <host> <port>
Run after `serve_policy.py` logs "listening", before launching eval workers."""
import sys, numpy as np
from openpi_client import msgpack_numpy as mn
import websockets.sync.client as ws

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
uri = f"ws://{host}:{port}"

# ping_interval=None: do NOT send websocket pings -> the connection survives the
# server's JAX compile (which blocks its loop and would otherwise miss pongs).
conn = ws.connect(uri, compression=None, max_size=None, ping_interval=None)
md = mn.unpackb(conn.recv())  # server metadata handshake
print(f"[warmup] connected to {uri}; sending dummy infer (JIT compile, may take 1-2min)...", flush=True)

z = np.zeros((3, 480, 640), dtype=np.uint8)
obs = {
    "images": {"cam_high": z, "cam_left_wrist": z, "cam_right_wrist": z},
    "state_arm": np.zeros(14, dtype=np.float32),
    "state_dex": np.zeros(44, dtype=np.float32),
    "prompt": "warmup",
}
conn.send(mn.Packer().pack(obs))
resp = conn.recv()  # blocks until compile done + action returned
print(f"[warmup] OK — server warm (response {len(resp)} bytes). Workers can connect now.", flush=True)
conn.close()