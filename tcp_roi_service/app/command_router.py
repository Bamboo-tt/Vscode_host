# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, Optional

Handler = Callable[[object, object], None]  # (conn, roi_service)


class CommandRouter:
    def __init__(self):
        self._routes: Dict[bytes, Handler] = {}
        self._unknown: Optional[Handler] = None  # 预留：你也可以自定义 unknown handler

    def register(self, cmd: bytes, handler: Handler) -> None:
        self._routes[cmd] = handler

    def set_unknown(self, handler: Handler) -> None:
        self._unknown = handler

    def dispatch(self, cmd: bytes, conn, roi_service) -> bool:
        """
        返回 True/False：
        - True：已处理（handler 执行完）
        - False：unknown cmd（tcp_server 会 break 并断开连接）
        """
        h = self._routes.get(cmd, self._unknown)
        if h is None:
            # 调试非常关键：能快速看出对端发了什么、是否流错位
            print(f"[TCP][UNKNOWN] cmd={cmd!r} hex={cmd.hex()}", flush=True)
            return False
        h(conn, roi_service)
        return True