# -*- coding: utf-8 -*-
from __future__ import annotations

from app.handlers._proto import send_ready, send_finish, recv_u8

"""
THR：设置灵敏度（0~100）
协议：
"THR" -> READY -> Sens(u8) -> FINISH

说明：
- handler 只负责读 sens 并调用 roi_service.set_sensitivity()
- conf_shm 的写入由 roi_service 内部完成（这样业务收口）
"""


def handle(conn, roi_service) -> None:
    send_ready(conn)

    sens = recv_u8(conn)  # 0~255，但业务里会 clamp 到 0~100
    roi_service.set_sensitivity(int(sens))

    send_finish(conn)