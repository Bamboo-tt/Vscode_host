# AI camera pipeline optimization notes

## Current finding

The upper-computer interface is already decoupled from the video pipeline:

- TCP service still listens on the existing port and commands.
- Detection results are still read from `/dev/shm/yolo_person_boxes`.
- Threshold updates are still written to `/dev/shm/yolo_conf_thr`.
- GPIO and model-upgrade paths are not tied to Python specifically.

That means the inference producer can be replaced by C++ as long as it writes the same 123-byte SHM layout:

```text
[0]    ready: uint8
[1:3]  count: uint16 little-endian
[3:]   10 * (x1,y1,x2,y2 uint16 + score uint8 + cls uint8 + pad uint16)
```

## Important issue

The camera exposes two V4L2 nodes from the same physical sensor:

- `/dev/video0`: main stream, used by RTSP.
- `/dev/video1`: sub stream, used by YOLO.

This is better than making YOLO pull RTSP on this board, because RTSP input had already proven unreliable and would also add decode latency. The important rule is still the same: do not make RTSP and YOLO both open `/dev/video0`.

## Recommended architecture

Best FPS and stability for the current Python deployment:

```text
/dev/video0 -> MPP H.265 encoder -> RTSP rtsp://<board>:8554/live
/dev/video1 -> RKNN inference -> same SHM output, scaled to main-stream coordinates
```

This keeps the hardware interface unchanged (`/dev/video0` and `/dev/video1`) and keeps the upper-computer interface unchanged (RTSP URL + TCP/SHM behavior), while avoiding RTSP decode in the YOLO path.

## Practical migration path

1. Keep the current Python services for compatibility.
2. Measure baseline with `journalctl -u yolo-to-shm.service -f` and the existing `[PERF]` log.
3. Avoid camera contention first:
   - RTSP owns `/dev/video0`.
   - YOLO owns `/dev/video1`.
   - Use `--out-width 1920 --out-height 1080` when YOLO captures a lower-resolution sub stream, so SHM boxes still match the main RTSP picture.
4. For best performance later, replace both Python video processes with one C++ GStreamer/RKNN process.
5. Move producer to C++:
   - V4L2/GStreamer capture in NV12.
   - Hardware resize/color conversion with RGA, not `cv2.resize` + `cv2.cvtColor`.
   - RKNN C API inference.
   - Existing C++ YOLOv8 postprocess logic reused directly, without pybind overhead.
   - Same `/dev/shm/yolo_person_boxes` writer.
6. Keep `tcp_roi_service` unchanged unless later you want to also port TCP/GPIO to C++.

## Fast settings to try now

The checked-in systemd services now use the dual-node layout:

```text
/dev/video0 -> rtsp-h265.service -> rtsp://<board>:8554/live
/dev/video1 -> yolo-to-shm.service -> /dev/shm/yolo_person_boxes
```

This keeps the upper-computer RTSP/TCP/SHM interfaces unchanged and avoids the failed RTSP-to-YOLO path.

For lower latency in the current Python producer:

```bash
/usr/local/bin/run_py310 /home/radxa/Security_monitoring/yolo_to_shm.py \
  --camera 1 \
  --width 1280 --height 720 --fps 30 \
  --out-width 1920 --out-height 1080 \
  --queue-size 1 --queue-drop-old \
  --size 640 --use-cpp-pp
```

If `/dev/video1` cannot be opened, first confirm the board really exposes the camera sub stream there with:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
```

On this board, YOLO pulling RTSP is not the preferred path because it had already failed in testing and would add decode latency even if it worked.
