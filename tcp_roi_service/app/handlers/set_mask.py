# -*- coding: utf-8 -*-
from __future__ import annotations

import struct

from app.handlers._proto import send_ready, send_finish, recv_u8
from app.tcp_server import recv_exact

"""
MSK：设置屏蔽矩形列表

语义：
- 检测框如果“完全包含于”(带 eps 容差)任意 MSK 内，则永远忽略（domain/rules.py 里实现）

协议：
"MSK" -> READY -> Num(u8) -> Num*(x1,y1,x2,y2 u16 BE) -> FINISH

约束：
- Num 最大 10
- Num=0 表示清空 mask
"""


MAX_MSK = 10


def handle(conn, roi_service) -> None:
    send_ready(conn)

    num = recv_u8(conn)
    if num > MAX_MSK:
        num = MAX_MSK

    masks = []
    if num > 0:
        payload = recv_exact(conn, num * 8)
        off = 0
        for _ in range(num):
            x1, y1, x2, y2 = struct.unpack_from("!HHHH", payload, off)
            off += 8
            masks.append((int(x1), int(y1), int(x2), int(y2)))

    roi_service.set_masks(masks)
    send_finish(conn)