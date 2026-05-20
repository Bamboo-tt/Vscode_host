# -*- coding: utf-8 -*-
from __future__ import annotations

import time


class AlarmHold:
    """
    贴合你源代码的 hold 行为：
    - set_triggered(True): 刷新 hold_until = now + hold_sec
    - output_now(): alarm_active 或者 hold 尚未结束
    """
    def __init__(self, hold_sec: float):
        self.hold_sec = float(hold_sec)
        self.alarm_active = False
        self.hold_until = 0.0

    def set_triggered(self, active: bool, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        self.alarm_active = bool(active)
        if self.alarm_active:
            self.hold_until = now + self.hold_sec

    def output_now(self, now: float | None = None) -> bool:
        if now is None:
            now = time.monotonic()
        return bool(self.alarm_active) or (now < float(self.hold_until))