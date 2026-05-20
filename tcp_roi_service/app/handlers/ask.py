# -*- coding: utf-8 -*-
from __future__ import annotations

import struct

"""
ASK：查询当前“有效报警框列表”
协议：
"ASK" -> Num(u8) + Num*(x1,y1,x2,y2 u16 BE)

说明：
- ASK 不走 READY/FINISH（与你原协议一致）
- 返回的 boxes 由 roi_service.get_alarm_boxes() 给出：
  已做 MSK 过滤、ROI 命中、去重、max_alarm 截断
"""


def handle(conn, roi_service) -> None:
    boxes = roi_service.get_alarm_boxes()

    num = len(boxes)
    if num > 255:
        num = 255

    conn.sendall(bytes([num]))

    if num > 0:
        payload = bytearray()
        for (x1, y1, x2, y2) in boxes[:num]:
            payload += struct.pack("!HHHH", int(x1), int(y1), int(x2), int(y2))
        conn.sendall(payload)