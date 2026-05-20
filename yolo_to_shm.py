#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_to_shm.py (Producer, threaded grabber)
作用：
- 采集摄像头帧 -> RKNNLite 推理 YOLOv8 -> 后处理(NMS/阈值) -> 写入共享内存 /dev/shm/yolo_person_boxes
- 支持动态阈值：读取 /dev/shm/yolo_conf_thr (float32 little-endian, 0~1)
- 支持后处理切换：C++ 扩展 / Python
- 支持性能计时打印：[PERF] pre / infer / pp / pack / shm / total + fps
- 采集线程与推理线程解耦：cap.read() 在后台线程持续拉帧，主线程专心推理与写 SHM
  * 默认队列满时丢旧帧（降低端到端延迟）
"""

import os
import sys
import time
import argparse
import mmap
import struct
import traceback
import threading
import queue
import signal

import cv2
import numpy as np

# 允许 import 同目录下的脚本/扩展模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------- SHM v2 布局 ----------------
# [0]    Ready: 1B
# [1:3]  Length: uint16 LE
# [3:]   10 * det(12B): x1y1x2y2(u16*4) + score(u8) + cls(u8) + pad(u16)
MAX_BOXES = 10
BOX_BYTES = 12
SHM_SIZE_V2 = 1 + 2 + MAX_BOXES * BOX_BYTES  # 123B

CONF_SHM_PATH_DEFAULT = "/dev/shm/yolo_conf_thr"
CONF_SHM_SIZE = 4  # float32


def parse_args():
    p = argparse.ArgumentParser(
        prog="yolo_to_shm.py",
        description="Camera -> RKNN YOLOv8 -> postprocess -> SHM v2 (threaded grabber)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # camera / shm
    p.add_argument("--source", default="", help="Override input source, e.g. /dev/video0 or rtsp://127.0.0.1:8554/live")
    p.add_argument("--fps", type=int, default=30, help="camera fps request; 0 means do not set")
    p.add_argument("--fourcc", default="", help="camera fourcc request, e.g. MJPG/YUYV/NV12; empty means do not set")
    p.add_argument("--camera", type=int, default=1, help="摄像头索引 /dev/video{camera}")
    p.add_argument("--width", type=int, default=1920, help="采集宽度；<=0 不设置")
    p.add_argument("--height", type=int, default=1080, help="采集高度；<=0 不设置")
    p.add_argument("--shm-path", default="/dev/shm/yolo_person_boxes", help="输出 SHM 路径(123B)")
    p.add_argument("--print-every", type=int, default=0, help="每隔N帧打印 [SHM]；0不打印")
    p.add_argument("--conf-shm", default=CONF_SHM_PATH_DEFAULT, help="动态阈值 SHM(float32 LE)")

    p.set_defaults(camera=0)

    # grab queue (new)
    p.add_argument("--queue-size", type=int, default=1, help="采集队列长度（越小延迟越低）")
    gq = p.add_mutually_exclusive_group()
    gq.add_argument("--queue-drop-old", dest="queue_drop_old", action="store_true", default=True,
                    help="队列满时丢旧帧（默认，延迟更低）")
    gq.add_argument("--queue-block", dest="queue_drop_old", action="store_false",
                    help="队列满时阻塞等待（不丢帧，但延迟可能更高）")

    # class filter
    g_cls = p.add_mutually_exclusive_group()
    g_cls.add_argument(
        "--person-only", dest="person_only", action="store_true", default=True,
        help="只输出 person-id 对应类别"
    )
    g_cls.add_argument(
        "--all-classes", dest="person_only", action="store_false",
        help="输出全部类别"
    )
    p.add_argument("--person-id", type=int, default=0, help="目标类别 id（单类通常=0）")

    # model / postprocess
    p.add_argument("--model", default="./model.rknn", help="RKNN 模型路径")
    p.add_argument("--size", type=int, default=640, help="模型输入尺寸 size x size")
    p.add_argument("--num-classes", type=int, default=80, help="类别数（单类=1，多类=80）")
    p.add_argument("--conf", type=float, default=0.50, help="默认 conf（读不到 conf_shm 才用）")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--topk", type=int, default=100, help="后处理候选上限")
    p.add_argument("--max-boxes", type=int, default=10, help="最多写入框数量（1~10）")
    p.add_argument(
        "--drop-if-busy", action="store_true",
        help="Ready=1(消费者未读)时丢本帧，不覆盖旧数据"
    )
    p.add_argument("--debug", action="store_true", help="打印推理输出 shape（配合 --print-every）")

    # nms mode
    g_nms = p.add_mutually_exclusive_group()
    g_nms.add_argument(
        "--class-agnostic-nms", dest="class_aware_nms",
        action="store_false", default=False,
        help="类无关 NMS（单类推荐）"
    )
    g_nms.add_argument(
        "--class-aware-nms", dest="class_aware_nms",
        action="store_true",
        help="类相关 NMS（多类可能需要）"
    )

    # postprocess backend（强制切换，便于 A/B 测速）
    g_pp = p.add_mutually_exclusive_group()
    g_pp.add_argument("--use-cpp-pp", dest="use_cpp_pp", action="store_true", default=True,
                      help="强制使用 C++ 后处理扩展（默认）")
    g_pp.add_argument("--use-py-pp", dest="use_cpp_pp", action="store_false",
                      help="强制使用 Python 后处理（用于对比测速）")

    # perf debug
    p.add_argument("--perf-every", type=int, default=30,
                   help="每隔N帧打印一次性能统计 [PERF]；0=关闭")
    p.add_argument("--perf-warmup", type=int, default=30,
                   help="前N帧不统计（预热后再计时）")
    p.add_argument("--perf-detail", action="store_true", default=True,
                   help="打印更详细的分段耗时（pre/infer/pp/pack/shm/total）")

    return p.parse_args()


def validate_args(a):
    if a.size <= 0:
        raise ValueError("--size 必须为正数")
    if not (0.0 <= a.conf <= 1.0):
        raise ValueError("--conf 必须在 0~1")
    if not (0.0 <= a.iou <= 1.0):
        raise ValueError("--iou 必须在 0~1")
    if not (1 <= a.max_boxes <= 10):
        raise ValueError("--max-boxes 必须在 1~10")
    if a.num_classes <= 0:
        raise ValueError("--num-classes 必须为正数")
    if a.topk <= 0:
        raise ValueError("--topk 必须为正数")
    if a.print_every < 0:
        raise ValueError("--print-every 不可为负")
    if a.perf_every < 0:
        raise ValueError("--perf-every 不可为负")
    if a.perf_warmup < 0:
        raise ValueError("--perf-warmup 不可为负")
    if a.queue_size <= 0:
        raise ValueError("--queue-size 必须为正数")


def clamp_u16(x: float) -> int:
    v = int(x)
    if v < 0:
        return 0
    if v > 65535:
        return 65535
    return v


def clamp_u8(x: int) -> int:
    v = int(x)
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


class ShmWriterV2:
    """作用：写 /dev/shm/yolo_person_boxes（SHM v2, 123B）"""

    def __init__(self, path: str):
        self.path = path
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, SHM_SIZE_V2)
        self.mm = mmap.mmap(fd, SHM_SIZE_V2, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.mm[:] = b"\x00" * SHM_SIZE_V2

    def write_dets(self, dets_u16_u8, drop_if_busy: bool) -> bool:
        """
        dets_u16_u8: list[(x1,y1,x2,y2, score_u8, cls_u8)]
        return: True写入成功；False丢帧
        """
        # 忙就丢帧：消费者还没读(Ready=1)则不覆盖
        if drop_if_busy and self.mm[0] == 1:
            return False

        # 允许覆盖写：为了避免消费者读到“半包”，先清 Ready=0 再写
        if (not drop_if_busy) and self.mm[0] == 1:
            self.mm[0:1] = b"\x00"

        n = min(len(dets_u16_u8), MAX_BOXES)

        # 写 Length
        self.mm[1:3] = struct.pack("<H", n)

        # 写 payload（不足补0）
        off = 3
        for i in range(MAX_BOXES):
            if i < n:
                x1, y1, x2, y2, sc, cl = dets_u16_u8[i]
                self.mm[off:off + 8] = struct.pack("<HHHH", x1, y1, x2, y2)
                self.mm[off + 8:off + 10] = bytes([sc, cl])
                self.mm[off + 10:off + 12] = struct.pack("<H", 0)
            else:
                self.mm[off:off + BOX_BYTES] = b"\x00" * BOX_BYTES
            off += BOX_BYTES

        # 最后置 Ready=1：表示本帧写完
        self.mm[0:1] = b"\x01"
        return True


class ConfShmReader:
    """作用：读 /dev/shm/yolo_conf_thr（float32 LE，0~1）"""

    def __init__(self, path: str):
        self.path = path
        self.mm = None

    def _ensure_open(self):
        if self.mm is not None:
            return
        if not os.path.exists(self.path):
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            self.mm = mmap.mmap(fd, CONF_SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)

    def read_conf(self):
        try:
            self._ensure_open()
            if self.mm is None:
                return None
            v = struct.unpack("<f", self.mm[0:4])[0]
            if not (0.0 <= float(v) <= 1.0):
                return None
            return float(v)
        except Exception:
            return None


def _configure_low_latency_source(dev: str) -> None:
    if not str(dev).lower().startswith("rtsp://"):
        return
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|max_delay;0|fflags;nobuffer|flags;low_delay",
    )


def _open_capture(dev: str) -> cv2.VideoCapture:
    _configure_low_latency_source(dev)
    if str(dev).startswith("/dev/video"):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(dev)


def open_camera(dev: str, width: int, height: int, fps: int, fourcc: str) -> cv2.VideoCapture:
    """作用：打开摄像头，并尽量设置到 1080p"""
    cap = _open_capture(dev)
    if not cap.isOpened():
        cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 {dev}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc).upper()[:4].ljust(4)))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def load_postprocess_backend(force_cpp: bool):
    """
    作用：
    - 选择后处理实现（C++ 扩展 / Python）
    - 返回 yolov8_rknn_post_process 函数
    """
    if force_cpp:
        import yolov8_postprocess_ext as _pp_mod
        fn = _pp_mod.yolov8_rknn_post_process
        print(f"[PP] using C++ ext: {_pp_mod.__file__}", flush=True)
        return fn
    else:
        import postprocess_yolov8_rknn as _pp_mod
        fn = _pp_mod.yolov8_rknn_post_process
        print(f"[PP] using Python: {_pp_mod.__file__}", flush=True)
        return fn


class FrameGrabber(threading.Thread):
    """
    作用：
    - 后台线程持续 cap.read() 拉帧
    - 写入队列供主线程推理
    - 默认队列满丢旧帧（降低延迟）
    """

    def __init__(self, cap: cv2.VideoCapture, q: queue.Queue, stop_event: threading.Event, drop_old: bool):
        super().__init__(daemon=True)
        self.cap = cap
        self.q = q
        self.stop_event = stop_event
        self.drop_old = drop_old

    def run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if (not ok) or (frame is None):
                continue

            if self.drop_old:
                # 队列满：丢掉最旧的（保持“最新帧优先”，降低延迟）
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try:
                        _ = self.q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        pass
            else:
                # 队列满：阻塞等待（不丢帧，但可能把延迟堆起来）
                while not self.stop_event.is_set():
                    try:
                        self.q.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        continue


def main():
    a = parse_args()
    validate_args(a)

    # 保留 stderr，异常时更容易在 systemd/journalctl 看到
    orig_stderr_fd = os.dup(2)

    stop_event = threading.Event()

    def _on_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    cap = None
    grabber = None

    try:
        from rknnlite.api import RKNNLite as RKNN

        # 选择后处理后端（便于 A/B 对比）
        yolov8_rknn_post_process = load_postprocess_backend(force_cpp=bool(a.use_cpp_pp))

        # camera
        cap_dev = str(a.source).strip() or f"/dev/video{a.camera}"
        cap = open_camera(cap_dev, a.width, a.height, int(a.fps), str(a.fourcc).strip())

        # 读首帧（确定分辨率与比例）
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            raise RuntimeError("读首帧失败")

        H, W = frame0.shape[:2]
        sx, sy = W / float(a.size), H / float(a.size)

        # shm + conf
        writer = ShmWriterV2(a.shm_path)
        conf_reader = ConfShmReader(a.conf_shm)

        # rknn init
        rknn = RKNN()
        if rknn.load_rknn(a.model) != 0:
            raise RuntimeError(f"加载RKNN失败: {a.model}")
        if rknn.init_runtime() != 0:
            raise RuntimeError("init_runtime 失败")

        # warmup（避免首帧抖动影响测速）
        img0 = cv2.resize(frame0, (a.size, a.size))
        rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        rknn.inference(inputs=[np.expand_dims(rgb0, 0).astype(np.uint8)])

        # 采集队列 + 采集线程
        q = queue.Queue(maxsize=int(a.queue_size))
        grabber = FrameGrabber(cap=cap, q=q, stop_event=stop_event, drop_old=bool(a.queue_drop_old))
        grabber.start()

        frame_id = 0
        last_conf = float(a.conf)

        while not stop_event.is_set():
            # 从队列取帧（包含等待时间）
            t_total0 = time.perf_counter()
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # preprocess
            t_pre0 = time.perf_counter()
            img = cv2.resize(frame, (a.size, a.size))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t_pre1 = time.perf_counter()

            # inference
            t_inf0 = time.perf_counter()
            outs = rknn.inference(inputs=[np.expand_dims(rgb, 0).astype(np.uint8)])
            t_inf1 = time.perf_counter()

            if a.debug and a.print_every > 0 and frame_id % max(1, a.print_every) == 0:
                shapes = [np.array(o).shape for o in outs]
                print(f"[DBG] outs_len={len(outs)} shapes={shapes}", flush=True)

            # dynamic conf
            dyn = conf_reader.read_conf()
            conf_thres = float(dyn) if dyn is not None else float(a.conf)
            conf_thres = 0.0 if conf_thres < 0.0 else (1.0 if conf_thres > 1.0 else conf_thres)

            # postprocess：C++ 扩展通常不支持关键字参数，统一用“位置参数”最稳
            t_pp0 = time.perf_counter()
            boxes, classes, scores = yolov8_rknn_post_process(
                outs,
                int(a.size),
                int(a.size),
                int(a.num_classes),
                float(conf_thres),
                float(a.iou),
                int(a.topk),
                bool(a.person_only),
                int(a.person_id),
                bool(getattr(a, "class_aware_nms", False)),
            )
            t_pp1 = time.perf_counter()

            # sort + pack
            t_pack0 = time.perf_counter()
            dets = []
            for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
                dets.append((float(s), float(x1), float(y1), float(x2), float(y2), int(c)))
            dets.sort(key=lambda t: t[0], reverse=True)
            dets = dets[: int(a.max_boxes)]

            out_dets = []
            for s, x1, y1, x2, y2, c in dets:
                sc_u8 = clamp_u8(int(round(float(s) * 255.0)))
                cl_u8 = clamp_u8(int(c))
                out_dets.append((
                    clamp_u16(x1 * sx), clamp_u16(y1 * sy),
                    clamp_u16(x2 * sx), clamp_u16(y2 * sy),
                    sc_u8, cl_u8
                ))
            t_pack1 = time.perf_counter()

            # shm write
            t_shm0 = time.perf_counter()
            wrote = writer.write_dets(out_dets, drop_if_busy=bool(a.drop_if_busy))
            t_shm1 = time.perf_counter()

            t_total1 = time.perf_counter()

            # periodic [SHM] print
            if a.print_every > 0 and (frame_id % a.print_every == 0):
                mode = "person-only" if a.person_only else "all-classes"
                changed = (abs(conf_thres - last_conf) > 1e-6)
                tag = "conf*" if changed else "conf"
                print(
                    f"[SHM] frame={frame_id} wrote={wrote} n={len(out_dets)} mode={mode} "
                    f"cap={cap_dev} res={W}x{H} size={a.size} {tag}={conf_thres:.2f}",
                    flush=True
                )
                last_conf = conf_thres

            # periodic [PERF] print
            if a.perf_every > 0 and frame_id >= int(a.perf_warmup) and (frame_id % int(a.perf_every) == 0):
                pre_ms = (t_pre1 - t_pre0) * 1000.0
                inf_ms = (t_inf1 - t_inf0) * 1000.0
                pp_ms = (t_pp1 - t_pp0) * 1000.0
                pack_ms = (t_pack1 - t_pack0) * 1000.0
                shm_ms = (t_shm1 - t_shm0) * 1000.0
                total_ms = (t_total1 - t_total0) * 1000.0
                fps = 1000.0 / total_ms if total_ms > 0 else 0.0

                if a.perf_detail:
                    print(
                        f"[PERF] f={frame_id} fps={fps:.2f} total={total_ms:.2f}ms "
                        f"pre={pre_ms:.2f} inf={inf_ms:.2f} pp={pp_ms:.2f} pack={pack_ms:.2f} shm={shm_ms:.2f} "
                        f"n={len(out_dets)} wrote={wrote} conf={conf_thres:.2f}",
                        flush=True
                    )
                else:
                    print(
                        f"[PERF] f={frame_id} fps={fps:.2f} total={total_ms:.2f}ms "
                        f"inf={inf_ms:.2f} pp={pp_ms:.2f} n={len(out_dets)}",
                        flush=True
                    )

            frame_id += 1

    except Exception:
        os.dup2(orig_stderr_fd, 2)
        traceback.print_exc()
        raise
    finally:
        stop_event.set()
        try:
            if grabber is not None and grabber.is_alive():
                grabber.join(timeout=1.0)
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            os.close(orig_stderr_fd)
        except Exception:
            pass


if __name__ == "__main__":
    main()
