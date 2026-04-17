#!/usr/bin/env python3
"""One-frame RGB smoke test for OAK-D (DepthAI v2 or v3).

Confirms the color camera path works without Zenoh or the full oakd_driver.
No OpenCV required for success; use -o only if you have opencv-python installed.

  .venv/bin/python scripts/oakd_rgb_smoke.py
  .venv/bin/python scripts/oakd_rgb_smoke.py -o /tmp/oakd_rgb.png
"""

from __future__ import annotations

import argparse
import sys

import depthai as dai


def _depthai_is_v3() -> bool:
    return not hasattr(dai.node, "XLinkOut")


def _one_frame_v3() -> dai.ImgFrame:
    # https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_output/
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        video_queue = cam.requestOutput((640, 400)).createOutputQueue()
        pipeline.start()
        frame = video_queue.get()
    assert isinstance(frame, dai.ImgFrame)
    return frame


def _one_frame_v2() -> dai.ImgFrame:
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 400)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xlink = pipeline.create(dai.node.XLinkOut)
    xlink.setStreamName("rgb")
    cam.preview.link(xlink.input)
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        frame = q.get()
    assert isinstance(frame, dai.ImgFrame)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Grab one RGB frame from OAK-D")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PNG",
        help="Save frame as PNG (requires: pip install opencv-python-headless)",
    )
    args = parser.parse_args()

    api = "v3" if _depthai_is_v3() else "v2"
    print(f"DepthAI API {api}; library {getattr(dai, '__version__', '?')}", flush=True)

    try:
        frame = _one_frame_v3() if _depthai_is_v3() else _one_frame_v2()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(
        f"OK: {frame.getWidth()}x{frame.getHeight()} type={frame.getType()} "
        f"bytes={len(frame.getData())}",
        flush=True,
    )

    if args.output:
        try:
            import cv2
        except ImportError:
            print(
                "PNG output needs OpenCV: pip install opencv-python-headless",
                file=sys.stderr,
            )
            return 2
        cv2.imwrite(args.output, frame.getCvFrame())
        print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
