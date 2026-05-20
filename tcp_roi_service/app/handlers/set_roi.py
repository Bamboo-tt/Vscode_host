# -*- coding: utf-8 -*-
from __future__ import annotations

import struct

from app.handlers._proto import send_ready, send_finish, recv_u8
from app.tcp_server import recv_exact

"""
SET：设置矩形 ROI 列表

协议：
"SET" -> READY -> Num(u8) -> Num * (x1,y1,x2,y2) (u16 BE) -> FINISH

约束：
- Num 最大 10（与 SHM v2 max_boxes 保持同量级，也符合你原脚本）
- handler 只做：协议收发 + struct 解包 + 调 roi_service.set_rect_rois()
"""


MAX_ROI = 10  # 可改成从 config/defaults 导入统一常量


def handle(conn, roi_service) -> None:
    # 1) 握手：告诉客户端可以开始发 payload
    send_ready(conn)

    # 2) 读取 ROI 数量（u8）
    num = recv_u8(conn)
    if num > MAX_ROI:
        num = MAX_ROI

    rois = []

    # 3) 按 num 解包，每个 rect 8 字节（!HHHH）
    if num > 0:
        payload = recv_exact(conn, num * 8)
        off = 0
        for _ in range(num):
            x1, y1, x2, y2 = struct.unpack_from("!HHHH", payload, off)
            off += 8
            rois.append((int(x1), int(y1), int(x2), int(y2)))

    # 4) 调业务中枢：更新 ROI，并由 service 内部触发重算报警/更新指示灯
    roi_service.set_rect_rois(rois)

    # 5) 回复完成
    send_finish(conn)