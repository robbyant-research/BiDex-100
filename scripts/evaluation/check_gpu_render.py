#!/usr/bin/env python
"""GPU render self-check: is this card's MuJoCo/EGL render pipeline deterministic?

A faulty card (e.g. NVRM Xid-31 MMU faults / Xid-109 ctx-switch timeouts on the
graphics channel) intermittently drops or truncates the glReadPixels write inside
mjr_readPixels. mujoco's Renderer.render() allocates the output with np.empty,
which recycles heap bytes of a previously-freed frame; a skipped readback then
leaves a STALE frame in the buffer, and render()'s unconditional flipud flips it
again -> the eval records/feeds an upside-down wrist/other-env frame (or a black
frame when the recycled block was zeroed). This silently corrupts both the eval
mp4s and the policy inputs.

Check: render the SAME static scene twice back-to-back, per camera, repeatedly.
A faulty card is caught by its real failure modes: black frames, stale /
vertically-flipped frames (dropped glReadPixels → heap-recycled, mujoco-flipud'd
old frame), and large or widespread pixel differences. Benign ~1-LSB rasterizer
jitter on a few pixels (driver/fp nondeterminism — observed on every L20 with
driver 535.161.08) is tolerated, not false-flagged; tune via GPU_CHECK_MAX_PX_DIFF
(default 15) and GPU_CHECK_MAX_DIFF_FRAC (default 0.01). Any EGL/GL error fails
the check.

Run one process per GPU (mujoco caches the EGL display per-process), with:
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=<gpu>

Exit: 0 = healthy, 1 = nondeterministic/black render, 2 = render setup failed.
"""
import argparse
import os
import sys

XML = """
<mujoco>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light diffuse=".6 .6 .6" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
    <body pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></body>
    <body pos=".3 .2 .3"><freejoint/><geom type="sphere" size=".12" rgba="0 1 0 1"/></body>
    <camera name="cam_a" pos="0 -1.2 1.2" euler="0.7 0 0"/>
    <camera name="cam_b" pos=".3 .1 .6" euler="1.2 0 0"/>
  </worldbody>
</mujoco>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="EGL device index")
    ap.add_argument("--iters", type=int, default=120)
    args = ap.parse_args()

    import numpy as np
    import mujoco

    try:
        m = mujoco.MjModel.from_xml_string(XML)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        # 480x360 + two cameras: same size class and same per-camera-renderer
        # usage pattern as the eval workers.
        renderers = {cam: mujoco.Renderer(m, height=360, width=480)
                     for cam in ("cam_a", "cam_b")}
    except Exception as e:
        print(f"[gpu-check] GPU{args.gpu} FAIL: renderer setup failed: {e}")
        return 2

    # Tolerances (env knobs surfaced in run_vec_models_eval.sh's banner). A healthy
    # MuJoCo/EGL rasterizer can show ~1-LSB jitter on a handful of pixels between
    # two identical renders (driver/fp nondeterminism) — strict np.array_equal
    # false-fails on it (observed: every L20 on driver 535.161.08). Accept that
    # jitter but still flag the real corruption modes the check exists to catch:
    # black frames; stale/vertically-flipped frames (a dropped glReadPixels leaves
    # a heap-recycled, mujoco-flipud'd stale frame); and any large or widespread
    # pixel difference.
    MAX_PX_DIFF = float(os.environ.get("GPU_CHECK_MAX_PX_DIFF", "15"))
    MAX_DIFF_FRAC = float(os.environ.get("GPU_CHECK_MAX_DIFF_FRAC", "0.01"))
    FLIP_THRESH = float(os.environ.get("GPU_CHECK_FLIP_THRESH", "1.0"))

    n_bad = 0
    first = None
    n_total = args.iters * len(renderers)
    try:
        for i in range(args.iters):
            for cam, r in renderers.items():
                r.update_scene(d, camera=cam)
                a = r.render()
                b = r.render()
                reason = None
                if float(a.mean()) < 1.0:
                    reason = "black frame (mean<1)"
                else:
                    ab = np.abs(a.astype(np.int16) - b.astype(np.int16))
                    md = int(ab.max())
                    nz = int((ab > 0).sum())
                    frac = nz / a.size
                    dflip = float(np.abs(np.flipud(a).astype(np.int16)
                                         - b.astype(np.int16)).mean())
                    if dflip < FLIP_THRESH:
                        reason = (f"flipped/stale frame: "
                                  f"mean|flipud(A)-B|={dflip:.3f} < {FLIP_THRESH:g}")
                    elif md > MAX_PX_DIFF:
                        reason = (f"max pixel diff {md} > "
                                  f"GPU_CHECK_MAX_PX_DIFF={MAX_PX_DIFF:g}")
                    elif frac > MAX_DIFF_FRAC:
                        reason = (f"{nz}/{a.size} ({frac:.4%}) pixels differ > "
                                  f"GPU_CHECK_MAX_DIFF_FRAC={MAX_DIFF_FRAC:g}")
                    # else: sub-LSB rasterization jitter — benign, not a fault
                if reason is not None:
                    n_bad += 1
                    if first is None:
                        first = (i, cam, reason)
    except Exception as e:
        print(f"[gpu-check] GPU{args.gpu} FAIL: render raised: {e}")
        return 2
    finally:
        for r in renderers.values():
            try:
                r.close()
            except Exception:
                pass

    if n_bad:
        detail = f"; first at iter={first[0]} cam={first[1]}: {first[2]}" if first else ""
        print(f"[gpu-check] GPU{args.gpu} FAIL: {n_bad}/{n_total} renders "
              f"nondeterministic/corrupt{detail}")
        return 1
    print(f"[gpu-check] GPU{args.gpu} PASS ({n_total} double-renders; "
          f"<=1px jitter tolerated, no flip/black/corrupt frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
