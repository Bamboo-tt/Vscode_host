#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rtsp_server_h265.py

作用：
- 在板端启动一个 RTSP Server（gst-rtsp-server）
- 从 /dev/videoX 取图（V4L2）
- 使用 Rockchip MPP 硬编（mpph265enc）编码为 H.265
- 通过 RTSP 输出给客户端（VLC/FFmpeg/RTSP Player）

客户端访问：
- rtsp://<板子IP>:8554/live   （默认端口 8554，默认挂载 /live）

关键设计点（非常重要）：
- gst-rtsp-server 的默认行为：只有“有客户端连接”时才创建 media pipeline，
  即：没有客户端连接时不会打开摄像头设备 /dev/videoX。
- 你的系统里常见情况：先启动推理/ISP（占用或重配），然后再连 RTSP，
  可能导致 RTSP 连接时 pipeline 无法正常创建（黑屏/失败）。
- 解决方案：本脚本默认启用“本地自连 keepalive”，
  启动后自动连接 rtsp://127.0.0.1:8554/live，迫使 pipeline 立即创建并常驻。

keepalive 开关（环境变量）：
- RTSP_KEEPALIVE=0           关闭 keepalive（默认开启）
- RTSP_KEEPALIVE_PROTO=tcp   keepalive 使用 tcp（默认 tcp；可改 udp）
- RTSP_KEEPALIVE_LATENCY=0   keepalive rtspsrc latency（默认 0）
- RTSP_KEEPALIVE_RESTART_SEC=5 watchdog 检查周期（默认 5 秒）

conda/gi 适配说明：
- 本脚本允许在 conda 环境运行（推荐 py310）
- 需要 conda 安装 pygobject：
    conda install -y -c conda-forge pygobject
- 需要系统安装 Gst/GstRtspServer typelibs：
    sudo apt install -y gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0
"""

import argparse
import os
import signal
import subprocess
import sys
from typing import Optional


# =============================================================================
# 1) GI / typelibs 适配（conda 环境下非常关键）
# =============================================================================
def _set_typelib_path():
    """
    作用：
    - pygobject(gi) 需要找到系统的 *.typelib（Gst / GstRtspServer）
    - conda 环境里经常找不到系统路径，所以这里把常见的 girepository 路径加入 GI_TYPELIB_PATH
    """
    cand = [
        "/usr/lib/aarch64-linux-gnu/girepository-1.0",
        "/usr/lib/arm-linux-gnueabihf/girepository-1.0",
        "/usr/lib/girepository-1.0",
        "/usr/lib64/girepository-1.0",
    ]
    exist = [p for p in cand if os.path.isdir(p)]
    if not exist:
        return

    joined = ":".join(exist)
    old = os.environ.get("GI_TYPELIB_PATH", "")

    # 只在未设置或不包含目标路径时追加，避免重复
    if old:
        if joined not in old:
            os.environ["GI_TYPELIB_PATH"] = old + ":" + joined
    else:
        os.environ["GI_TYPELIB_PATH"] = joined


def _die(msg: str, code: int = 2):
    """作用：统一的错误退出函数（把信息输出到 stderr 并退出）"""
    sys.stderr.write(msg.rstrip() + "\n")
    raise SystemExit(code)


_set_typelib_path()

try:
    import gi
except Exception as e:
    _die(
        "[ERR] 当前 conda 环境缺 gi(Pygobject)。\n"
        "在 conda 环境执行：\n"
        "  conda install -y -c conda-forge pygobject\n"
        f"当前 Python: {sys.executable}\n"
        f"原始错误: {repr(e)}"
    )

try:
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
except Exception as e:
    _die(
        "[ERR] 找不到 Gst/GstRtspServer 的命名空间（typelibs 缺失/不可用）。\n"
        "执行：\n"
        "  sudo apt update\n"
        "  sudo apt install -y gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 libgstrtspserver-1.0-0\n"
        f"GI_TYPELIB_PATH={os.environ.get('GI_TYPELIB_PATH','')}\n"
        f"原始错误: {repr(e)}"
    )

from gi.repository import Gst, GstRtspServer, GLib  # noqa: E402


# =============================================================================
# 2) keepalive（自连自保活）：强制 pipeline 立即创建并常驻
# =============================================================================
def _keepalive_enabled() -> bool:
    """作用：读取环境变量，决定是否启用 keepalive（默认启用）"""
    return os.environ.get("RTSP_KEEPALIVE", "1") not in ("0", "false", "False")


def _keepalive_latency() -> int:
    """作用：keepalive 的 rtspsrc latency 参数（默认 0）"""
    try:
        return int(os.environ.get("RTSP_KEEPALIVE_LATENCY", "0"))
    except Exception:
        return 0


def _keepalive_protocol() -> str:
    """
    作用：
    - keepalive 连接使用的协议（tcp/udp）
    - rtspsrc 的 protocols 字段可接受 tcp/udp
    """
    v = os.environ.get("RTSP_KEEPALIVE_PROTO", "tcp").strip().lower()
    return "tcp" if v not in ("udp",) else "udp"


def _keepalive_restart_sec() -> int:
    """作用：watchdog 检查周期（秒），默认 5，最小 2"""
    try:
        v = int(os.environ.get("RTSP_KEEPALIVE_RESTART_SEC", "5"))
        return 2 if v < 2 else v
    except Exception:
        return 5


class _KeepAlive:
    """
    作用：
    - 本地“自连”到 RTSP server，确保 media factory 的 pipeline 立即创建
    - 优先使用 Python/Gst 直接建 pipeline
    - 若插件缺失，则 fallback 用 gst-launch 子进程
    """

    def __init__(self, url: str, latency: int, proto: str):
        self.url = url
        self.latency = latency
        self.proto = proto
        self.pipeline: Optional[Gst.Element] = None
        self.proc: Optional[subprocess.Popen] = None

    def _can_use_python_gst(self) -> bool:
        """
        作用：检查 keepalive 所需插件是否存在
        - rtspsrc：拉流
        - rtph265depay：RTP 解包
        - h265parse：解析
        - fakesink：丢弃输出（只为维持连接）
        """
        need = ("rtspsrc", "rtph265depay", "h265parse", "fakesink")
        return all(Gst.ElementFactory.find(x) is not None for x in need)

    def start(self):
        """
        作用：启动 keepalive
        - 若已有 keepalive，则先 stop 再重启
        """
        self.stop()

        # 方案 A：用 Python/Gst（更干净，无子进程）
        if self._can_use_python_gst():
            launch = (
                f"rtspsrc location={self.url} latency={self.latency} protocols={self.proto} ! "
                "rtph265depay ! h265parse ! fakesink sync=false"
            )
            try:
                self.pipeline = Gst.parse_launch(launch)
                self.pipeline.set_state(Gst.State.PLAYING)
                print(f"[KEEP] ON (python) url={self.url}", flush=True)
                return
            except Exception as e:
                self.pipeline = None
                print(f"[KEEP] WARN: python keepalive failed: {repr(e)}", flush=True)

        # 方案 B：fallback 用 gst-launch（插件不足或 parse_launch 失败时兜底）
        try:
            cmd = (
                "gst-launch-1.0 -q "
                f"rtspsrc location='{self.url}' latency={self.latency} protocols={self.proto} ! "
                "rtph265depay ! h265parse ! fakesink sync=false"
            )
            self.proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[KEEP] ON (gst-launch) url={self.url}", flush=True)
        except Exception as e:
            self.proc = None
            print(f"[KEEP] WARN: gst-launch keepalive failed: {repr(e)}", flush=True)

    def stop(self):
        """作用：停止 keepalive（关闭 pipeline 或杀掉 gst-launch 子进程）"""
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None

        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

    def healthy(self) -> bool:
        """
        作用：判断 keepalive 是否健康
        - Python/Gst：状态是否 PLAYING
        - gst-launch：子进程是否仍存活
        """
        if self.pipeline is not None:
            try:
                _, state, _ = self.pipeline.get_state(0)
                return state == Gst.State.PLAYING
            except Exception:
                return False
        if self.proc is not None:
            return self.proc.poll() is None
        return False


# =============================================================================
# 3) 工具函数：IP 获取 / pipeline 构造 / 参数解析
# =============================================================================
def guess_board_ip(default_ip: str) -> str:
    """
    作用：尽可能自动获取板子的出口 IPv4（用于打印 Client URL）
    方法：ip route get 1.1.1.1 取 src 字段
    失败：返回 default_ip
    """
    try:
        out = subprocess.check_output(
            ["bash", "-lc",
             "ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i==\"src\"){print $(i+1); exit}}'"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return out or default_ip
    except Exception:
        return default_ip


def build_pipeline(
    device: str,
    io_mode: int,
    width: int,
    height: int,
    fps: int,
    pixfmt: str,
    bitrate: int,
    gop: int,
    rc_mode: str,
    parse_config_interval: int,
    pay_config_interval: int,
) -> str:
    """
    作用：构造 RTSP MediaFactory 使用的 GStreamer launch 字符串

    注意点：
    - rtph265pay 必须 name=pay0（gst-rtsp-server 识别用）
    - pay 的 config-interval 建议 1（客户端中途加入时更容易拿到 SPS/PPS/VPS）
    - h265parse 的 config-interval 一般保持默认即可（-1）
    """
    return (
        # 摄像头输入（V4L2）
        f"v4l2src device={device} io-mode={io_mode} do-timestamp=true ! "
        # 强制 caps（分辨率/帧率/像素格式必须是驱动支持的）
        f"video/x-raw,format={pixfmt},width={width},height={height},framerate={fps}/1 ! "
        "queue max-size-buffers=2 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
        # Rockchip MPP H.265 硬编码
        f"mpph265enc bps={bitrate} gop={gop} rc-mode={rc_mode} ! "
        # 解析/打包
        f"h265parse config-interval={parse_config_interval} ! "
        f"rtph265pay name=pay0 pt=96 config-interval={pay_config_interval}"
    )


def parse_args():
    """作用：定义命令行参数（尽量保持和你原来用法一致）"""
    p = argparse.ArgumentParser(description="Rockchip V4L2 -> MPP H265 -> RTSP Server (GStreamer)")

    # RTSP 服务参数
    p.add_argument("--bind", default="0.0.0.0", help="RTSP 监听地址（建议 0.0.0.0）")
    p.add_argument("--port", type=int, default=8554, help="RTSP 端口")
    p.add_argument("--mount", default="/live", help="RTSP 挂载路径，如 /live")

    # 摄像头/视频参数
    p.add_argument("--device", default="/dev/video0", help="摄像头节点，如 /dev/video0")
    p.add_argument("--io-mode", type=int, default=4, help="v4l2src io-mode（常用 2/4，按驱动支持）")
    p.add_argument("--width", type=int, default=1920, help="宽度")
    p.add_argument("--height", type=int, default=1080, help="高度")
    p.add_argument("--fps", type=int, default=30, help="帧率（必须是驱动支持的值）")
    p.add_argument("--pixfmt", default="NV12", help="像素格式（常用 NV12/YUY2 等）")

    # 编码参数
    p.add_argument("--bitrate", type=int, default=6000000, help="码率 bps（如 6000000=6Mbps）")
    p.add_argument("--gop", type=int, default=60, help="GOP（关键帧间隔）")
    p.add_argument("--rc-mode", default="cbr", choices=["cbr", "vbr"], help="码率控制模式")

    # 兼容性参数
    p.add_argument(
        "--h265parse-config-interval",
        type=int,
        default=-1,
        help="h265parse config-interval（默认 -1）",
    )
    p.add_argument(
        "--pay-config-interval",
        type=int,
        default=1,
        help="rtph265pay config-interval（建议 1）",
    )

    # 诊断
    p.add_argument("--gst-debug", default="", help="设置 GST_DEBUG（例如 2 或 3 或 '*:2'）")

    return p.parse_args()


def validate_args(a):
    """作用：对参数做基本校验，避免明显错误导致 gst 启动失败"""
    if not a.mount.startswith("/"):
        _die("[ERR] --mount 必须以 / 开头，例如 /live")
    if a.fps <= 0 or a.width <= 0 or a.height <= 0:
        _die("[ERR] width/height/fps 必须为正数")
    if a.bitrate < 100_000:
        sys.stderr.write("[WARN] bitrate 太低，可能影响画质或码控稳定性\n")


# =============================================================================
# 4) RTSP MediaFactory：负责创建 pipeline（gst-rtsp-server 会在需要时调用）
# =============================================================================
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    """
    作用：
    - 这是 gst-rtsp-server 的媒体工厂
    - 当有客户端访问 mount path 时，server 会通过它创建媒体 pipeline
    """

    def __init__(self, pipeline: str):
        super().__init__()
        self._pipeline = pipeline
        # 作用：多个客户端共享同一路编码管道（只编码一次，多个客户端复用）
        self.set_shared(True)

    def do_create_element(self, url):
        # 作用：创建实际的 pipeline 元素
        print(f"[RTSP] Creating pipeline:\n{self._pipeline}\n", flush=True)
        return Gst.parse_launch(self._pipeline)


# =============================================================================
# 5) 主逻辑：启动 RTSP Server + keepalive + 主循环
# =============================================================================
def main():
    args = parse_args()
    validate_args(args)

    # 作用：可选打开 GStreamer 调试（排查 pipeline 问题非常有用）
    if args.gst_debug:
        os.environ["GST_DEBUG"] = args.gst_debug

    # 作用：打印 URL 时用的 IP（不影响实际监听）
    board_ip = guess_board_ip("192.168.88.239")

    # 作用：初始化 GStreamer
    Gst.init(None)

    # 作用：提前检查关键插件，缺了就直接报错（避免跑到一半才失败）
    if Gst.ElementFactory.find("mpph265enc") is None:
        _die("[ERR] 找不到 mpph265enc 插件（Rockchip MPP 编码器）。先确认 gstreamer/mpp 插件已安装。", 1)

    # 作用：构造 “V4L2 -> MPP H265 -> RTP -> RTSP” pipeline
    pipeline = build_pipeline(
        device=args.device,
        io_mode=args.io_mode,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixfmt=args.pixfmt,
        bitrate=args.bitrate,
        gop=args.gop,
        rc_mode=args.rc_mode,
        parse_config_interval=args.h265parse_config_interval,
        pay_config_interval=args.pay_config_interval,
    )

    # 作用：创建 RTSP Server，并绑定监听地址/端口
    server = GstRtspServer.RTSPServer()
    server.set_address(args.bind)
    server.set_service(str(args.port))

    # 作用：挂载媒体工厂到指定路径
    mounts = server.get_mount_points()
    factory = SensorFactory(pipeline)
    mounts.add_factory(args.mount, factory)

    # 作用：启动 server（attach 成功返回非 0）
    if server.attach(None) == 0:
        _die("[ERR] RTSP server attach failed.", 1)

    print("\n" + "=" * 60)
    print("[OK] RTSP server started")
    print(f"Listen on : {args.bind}:{args.port}")
    print(f"Mount     : {args.mount}")
    print(f"Client URL: rtsp://{board_ip}:{args.port}{args.mount}")
    print("=" * 60 + "\n")

    # 作用：GLib 主循环（gst-rtsp-server 基于 GLib 事件循环运行）
    loop = GLib.MainLoop()

    # -------------------------------------------------------------------------
    # keepalive：本地自连，强制 pipeline 立即创建（默认开启；RTSP_KEEPALIVE=0 可关）
    # -------------------------------------------------------------------------
    keep = None
    if _keepalive_enabled():
        # 作用：用 127.0.0.1 访问本机 RTSP，避免走外网/路由
        local_url = f"rtsp://127.0.0.1:{args.port}{args.mount}"
        keep = _KeepAlive(local_url, _keepalive_latency(), _keepalive_protocol())

        def _start_keepalive_once():
            # 作用：延迟一点点再启动，给 server attach 留出时间
            keep.start()
            return False  # 只执行一次

        def _keepalive_watchdog():
            # 作用：周期性检查 keepalive 是否还活着，死了就重启
            if keep is not None and not keep.healthy():
                keep.start()
            return True

        # 300ms 后启动一次 keepalive
        GLib.timeout_add(300, _start_keepalive_once)
        # 每 N 秒检查一次 keepalive
        GLib.timeout_add_seconds(_keepalive_restart_sec(), _keepalive_watchdog)

    # 作用：优雅退出（systemd stop / Ctrl+C）
    def _stop(*_):
        print("\n[RTSP] Stopping...", flush=True)
        if keep is not None:
            keep.stop()
        loop.quit()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # 作用：进入主循环（阻塞）
    loop.run()

    # 作用：主循环退出后再确保 keepalive 停止
    if keep is not None:
        keep.stop()


if __name__ == "__main__":
    main()
