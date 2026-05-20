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

Do not run two independent processes that both open `/dev/video0` unless the camera driver is proven to support it.

The current RTSP server opens `/dev/video0` directly. If `yolo_to_shm.py` is also changed to `--camera 0`, both processes compete for the same V4L2 node. On RK3566 this often causes lower FPS, black frames, long latency, or unstable startup.

## Recommended architecture

Best FPS and stability:

```text
/dev/video0
  -> one capture owner
  -> tee
     -> MPP H.265 encoder -> RTSP rtsp://<board>:8554/live
     -> RGA resize/color convert -> RKNN inference -> same SHM output
```

This keeps the hardware interface unchanged (`/dev/video0`) and keeps the upper-computer interface unchanged (RTSP URL + TCP/SHM behavior), but removes duplicate camera access and extra frame copies.

## Practical migration path

1. Keep the current Python services for compatibility.
2. Measure baseline with `journalctl -u yolo-to-shm.service -f` and the existing `[PERF]` log.
3. Avoid camera contention first:
   - If RTSP must own `/dev/video0`, let YOLO read `--source rtsp://127.0.0.1:8554/live` as a compatibility test.
   - For best performance, replace both Python video processes with one C++ GStreamer/RKNN process.
4. Move producer to C++:
   - V4L2/GStreamer capture in NV12.
   - Hardware resize/color conversion with RGA, not `cv2.resize` + `cv2.cvtColor`.
   - RKNN C API inference.
   - Existing C++ YOLOv8 postprocess logic reused directly, without pybind overhead.
   - Same `/dev/shm/yolo_person_boxes` writer.
5. Keep `tcp_roi_service` unchanged unless later you want to also port TCP/GPIO to C++.

## Fast settings to try now

The checked-in systemd services now use the safer compatibility layout:

```text
/dev/video0 -> rtsp-h265.service -> rtsp://127.0.0.1:8554/live -> yolo-to-shm.service
```

This keeps `/dev/video0` as the hardware input and keeps the upper-computer RTSP/TCP/SHM interfaces unchanged, while avoiding two independent processes opening the camera node at the same time.

For lower latency in the current Python producer:

```bash
/usr/local/bin/run_py310 /home/radxa/Security_monitoring/yolo_to_shm.py \
  --source /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --queue-size 1 --queue-drop-old \
  --size 640 --use-cpp-pp
```

If RTSP is already using `/dev/video0`, do not use the command above at the same time. Use this compatibility mode instead:

```bash
/usr/local/bin/run_py310 /home/radxa/Security_monitoring/yolo_to_shm.py \
  --source rtsp://127.0.0.1:8554/live \
  --width 0 --height 0 --fps 0 \
  --queue-size 1 --queue-drop-old \
  --size 640 --use-cpp-pp
```

The RTSP-read mode avoids V4L2 contention but adds H.265 decode cost, so it is a compatibility fix, not the final highest-FPS design.
