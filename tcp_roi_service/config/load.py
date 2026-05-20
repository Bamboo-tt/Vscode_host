# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from config import defaults as D


@dataclass
class Config:
    """
    运行配置对象（由 argparse 解析得到）。
    设计原则：
    - 所有运行时参数集中在这里，main.py 只拿 Config 去初始化各模块
    - 默认值统一来自 config/defaults.py（避免散落在各文件）
    """

    # ===== TCP Server 参数 =====
    bind: str              # 监听地址，例如 "0.0.0.0"
    port: int              # 监听端口，例如 9000
    client_timeout: int    # 单连接 socket 超时（秒），避免僵尸连接

    # ===== SHM 路径 =====
    shm_path: str          # 检测框共享内存（Producer 写，TCP/ROI 服务读）
    conf_shm: str          # 动态阈值共享内存（TCP/ROI 服务写，Producer 读）

    # ===== UPG 升级相关 =====
    model_path: str        # 目标模型文件路径（UPG 接收到的新模型会替换到这里）
    restart_unit: str      # UPG 完成后需要重启的 systemd unit 名（例如 yolo-to-shm.service）
    max_model_bytes: int   # UPG 接收的最大模型大小限制，防止误传/DoS

    # ===== ROI/报警业务参数 =====
    poll_ms: int               # 轮询 SHM 的周期（毫秒）
    max_alarm: int             # ASK 返回的最大报警框数量（也用于去重后截断）
    default_sensitivity: int   # 启动时默认灵敏度 0~100（THR 未设置前用这个）
    msk_eps: int               # MSK 容差像素：判定“框是否完全落在 mask 内”时扩张/容忍

    # ===== 调试打印开关 =====
    print_shm: bool        # 打印 SHM 读到的 det 数量/阈值等
    print_roi: bool        # 打印 SET/POL（ROI 更新）
    print_thr: bool        # 打印 THR（阈值更新）
    print_msk: bool        # 打印 MSK（mask 更新）


def load_config(argv: list[str] | None = None) -> Config:
    """
    解析命令行参数并返回 Config。

    - argv=None：默认从 sys.argv 读取（正常运行）
    - argv=list：便于单元测试/脚本调用时自定义参数

    说明：
    - 这里不做“业务校验”（例如 msk_eps 合法性），校验/Clamp 放在 RoiService 内更合理
    - 默认值全部来自 config/defaults.py，便于统一维护与部署
    """
    p = argparse.ArgumentParser()

    # ===== TCP 参数 =====
    p.add_argument("--bind", default=D.DEFAULT_BIND,
                   help="TCP 监听地址，例如 0.0.0.0")
    p.add_argument("--port", type=int, default=D.DEFAULT_PORT,
                   help="TCP 监听端口，例如 9000")
    p.add_argument("--client-timeout", type=int, default=D.DEFAULT_CLIENT_TIMEOUT,
                   help="单个客户端连接的 socket 超时（秒），超时则断开连接")

    # ===== SHM 路径 =====
    p.add_argument("--shm-path", default=D.DEFAULT_SHM_PATH,
                   help="检测框 SHM 路径（Producer 写入）")
    p.add_argument("--conf-shm", default=D.DEFAULT_CONF_SHM,
                   help="动态阈值 SHM 路径（本服务写入，Producer 读取）")

    # ===== 业务参数 =====
    p.add_argument("--poll-ms", type=int, default=D.DEFAULT_POLL_MS,
                   help="轮询 SHM 的周期（毫秒）")
    p.add_argument("--max-alarm", type=int, default=D.DEFAULT_MAX_ALARM,
                   help="ASK 返回的最大报警框数量（去重后截断）")
    p.add_argument("--default-sensitivity", type=int, default=D.DEFAULT_DEFAULT_SENSITIVITY,
                   help="启动默认灵敏度 0~100（THR 未设置前使用）")
    p.add_argument("--msk-eps", type=int, default=D.DEFAULT_MSK_EPS,
                   help="MSK 容差像素（判定框在 mask 内的容忍/扩张）")

    # ===== UPG 参数 =====
    p.add_argument("--model-path", default=D.DEFAULT_MODEL_PATH,
                   help="UPG 接收模型后替换到该路径")
    p.add_argument("--restart-unit", default=D.DEFAULT_RESTART_UNIT,
                   help="UPG 成功后重启的 systemd unit 名")
    p.add_argument("--max-model-bytes", type=int, default=D.MAX_MODEL_BYTES,
                   help="UPG 最大允许模型大小（字节）")

    # ===== 调试打印 =====
    # 说明：store_true 表示默认 False，传了该参数就变 True
    p.add_argument("--print-shm", action="store_true", default=D.DEFAULT_PRINT_SHM,
                   help="打印 SHM 读取状态（调试用）")
    p.add_argument("--print-roi", action="store_true", default=D.DEFAULT_PRINT_ROI,
                   help="打印 ROI 更新（SET/POL）")
    p.add_argument("--print-thr", action="store_true", default=D.DEFAULT_PRINT_THR,
                   help="打印阈值更新（THR）")
    p.add_argument("--print-msk", action="store_true", default=D.DEFAULT_PRINT_MSK,
                   help="打印 mask 更新（MSK）")

    # 开始解析参数（argv=None 则用 sys.argv）
    a = p.parse_args(argv)

    # 构造 Config 返回：main.py 只拿 Config 去初始化服务
    return Config(
        bind=a.bind,
        port=int(a.port),
        client_timeout=int(a.client_timeout),

        shm_path=a.shm_path,
        conf_shm=a.conf_shm,

        model_path=a.model_path,
        restart_unit=a.restart_unit,
        max_model_bytes=int(a.max_model_bytes),

        poll_ms=int(a.poll_ms),
        max_alarm=int(a.max_alarm),
        default_sensitivity=int(a.default_sensitivity),
        msk_eps=int(a.msk_eps),
 
        print_shm=bool(a.print_shm),
        print_roi=bool(a.print_roi),
        print_thr=bool(a.print_thr),
        print_msk=bool(a.print_msk),
    )