# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import threading
from typing import Optional

from config.defaults import BEEP_GPIO, LEDB_GPIO, LEDR_GPIO, LEDG_GPIO, LED_HOLD_SEC, ALARM_HOLD_SEC


SYSFS_BASE = "/sys/class/gpio"


class SysfsGpioPin:
    def __init__(self, gpio: int, name: str):
        self.gpio = int(gpio)
        self.name = str(name)
        self.gpio_dir = os.path.join(SYSFS_BASE, f"gpio{self.gpio}")
        self.value_path = os.path.join(self.gpio_dir, "value")
        self.direction_path = os.path.join(self.gpio_dir, "direction")

    @staticmethod
    def _write_text(path: str, s: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)

    def export(self) -> None:
        if os.path.isdir(self.gpio_dir):
            return
        export_path = os.path.join(SYSFS_BASE, "export")
        try:
            self._write_text(export_path, str(self.gpio))
        except OSError:
            pass
        for _ in range(50):
            if os.path.isdir(self.gpio_dir):
                return
            time.sleep(0.01)

    def ensure_out(self) -> None:
        self.export()
        if not os.path.isdir(self.gpio_dir):
            raise RuntimeError(f"GPIO {self.gpio} export failed")
        self._write_text(self.direction_path, "out")

    def write(self, v: int) -> None:
        self._write_text(self.value_path, "1" if int(v) else "0")


class IndicatorController:
    MODE_OK = "ok"
    MODE_ERR = "err"

    def __init__(self, beep_on_err: bool = True):
        self.beep = SysfsGpioPin(BEEP_GPIO, "BEEP")
        self.ledb = SysfsGpioPin(LEDB_GPIO, "LEDB")
        self.ledr = SysfsGpioPin(LEDR_GPIO, "LEDR")
        self.ledg = SysfsGpioPin(LEDG_GPIO, "LEDG")

        self._mode: Optional[str] = None
        self._hold_until: float = 0.0
        self._err_hold_until: float = 0.0
        self._lock = threading.Lock()
        self._beep_on_err = bool(beep_on_err)

    def init(self):
        for p in (self.beep, self.ledb, self.ledr, self.ledg):
            p.ensure_out()
        self.force_mode(self.MODE_OK)

    def _set_led(self, r: int, g: int, b: int):
        self.ledr.write(r)
        self.ledg.write(g)
        self.ledb.write(b)

    @staticmethod
    def _prio(mode: str) -> int:
        return 2 if mode == IndicatorController.MODE_ERR else 1

    def force_mode(self, mode: str):
        with self._lock:
            self._apply_mode_locked(mode, force=True)

    def set_mode(self, mode: str):
        with self._lock:
            self._apply_mode_locked(mode, force=False)

    def _apply_mode_locked(self, mode: str, force: bool):
        now = time.monotonic()
        if mode == self._mode:
            return

        if (not force) and (self._mode == self.MODE_ERR) and (mode == self.MODE_OK):
            if now < self._err_hold_until:
                return

        if (not force) and (now < self._hold_until) and (self._mode is not None):
            if self._prio(mode) <= self._prio(self._mode):
                return

        if mode == self.MODE_OK:
            self._set_led(r=0, g=1, b=0)
            self.beep.write(0)
        elif mode == self.MODE_ERR:
            self._set_led(r=1, g=0, b=0)
            self.beep.write(1 if self._beep_on_err else 0)
            self._err_hold_until = now + ALARM_HOLD_SEC
        else:
            raise ValueError(f"unknown mode: {mode}")

        self._mode = mode
        self._hold_until = now + LED_HOLD_SEC