# -*- coding: utf-8 -*-
from __future__ import annotations

from app.handlers._proto import send_ready, send_finish, send_fail, recv_u32_be
from app.tcp_server import recv_exact

"""
UPG：模型升级（接收 rknn 文件并替换）

协议（与你原脚本一致）：
"UPG" -> READY -> Len(u32 BE) + <Len bytes> -> FINISH -> (后台 restart)

语义：
- FINISH 表示：文件已接收且写盘/替换成功
- 重启在后台执行，不阻塞 TCP；重启失败不影响 FINISH
- FAIL 仅表示：接收/校验/写盘替换失败
"""


def handle(conn, roi_service) -> None:
    send_ready(conn)

    length = recv_u32_be(conn)

    # 取 max_model_bytes：优先用 roi_service 上的配置；没有就退化到一个安全默认
    max_bytes = getattr(roi_service, "max_model_bytes", 200 * 1024 * 1024)

    # 与原版一致：严格校验长度，避免被超大 length 卡死/DoS
    if length <= 0 or length > int(max_bytes):
        send_fail(conn)
        return

    # 读文件内容
    data = recv_exact(conn, int(length))

    try:
        # roi_service.upgrade_model 内部应做：
        # 1) 原子替换文件（写盘成功才算成功）
        # 2) 先返回（handler 回 FINISH）
        # 3) 后台线程 restart 推理服务
        roi_service.upgrade_model(data)

        # 写盘替换成功 -> FINISH（上位机无需等待重启完成）
        send_finish(conn)

    except Exception:
        send_fail(conn)