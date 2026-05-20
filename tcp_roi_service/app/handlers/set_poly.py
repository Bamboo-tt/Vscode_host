# -*- coding: utf-8 -*-
from __future__ import annotations

import struct
from typing import List, Tuple

from app.handlers._proto import send_ready, send_finish, recv_u8
from app.tcp_server import recv_exact

"""
POL：设置多边形 ROI 列表（变长）

协议：
"POL" -> READY -> Num(u8)
       -> for each poly:
            Vcnt(u8) + Vcnt*(x,y) (u16 BE)
       -> FINISH

约束（对齐你原脚本习惯）：
- Num 最大 10
- 每个 polygon 顶点数 Vcnt 最大 32
- Vcnt < 3 视为无效 polygon：但仍要把对应字节消费掉（避免流错位）
"""


MAX_POLY = 10
MAX_VERT = 32
Point = Tuple[int, int]
Polygon = List[Point]


def _recv_polylists(conn, num: int) -> List[Polygon]:
    polys: List[Polygon] = []

    for _ in range(num):
        vcnt = recv_u8(conn)
        if vcnt > MAX_VERT:
            vcnt = MAX_VERT

        # vcnt<3：无效；但如果 vcnt>0 仍需消费 vcnt*4 字节
        if vcnt < 3:
            if vcnt > 0:
                recv_exact(conn, vcnt * 4)
            continue

        raw = recv_exact(conn, vcnt * 4)
        pts: Polygon = []
        off = 0
        for _ in range(vcnt):
            x, y = struct.unpack_from("!HH", raw, off)
            off += 4
            pts.append((int(x), int(y)))

        polys.append(pts)

    return polys


def handle(conn, roi_service) -> None:
    send_ready(conn)

    num = recv_u8(conn)
    if num > MAX_POLY:
        num = MAX_POLY

    polys = _recv_polylists(conn, num) if num > 0 else []

    roi_service.set_poly_rois(polys)
    send_finish(conn)