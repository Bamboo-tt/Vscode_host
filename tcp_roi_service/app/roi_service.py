# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import threading
import subprocess
from typing import List, Optional, Tuple

from config.defaults import ALARM_HOLD_SEC
from domain.types import Box, Det, Polygon, Rect
from domain.geometry import aabb_intersects, rect_poly_intersects, rect_contains

from adapters.shm_reader import ShmReader
from adapters.conf_shm import ConfShmWriter
from adapters.gpio import IndicatorController


class RoiService:
    """
    等价迁移自你原 tcp_roi_server_shm.py 的 RoiAlarmServer（去掉 TCP handlers）：
    - 后台线程轮询 SHM：更新 dets -> 计算报警 -> 更新 hold -> 刷新指示灯
    - THR：conf = sens/100.0 写入 conf_shm
    - 无 ROI 直接不报警（与你原版一致）
    - UPG：atomic replace + 后台 systemctl restart --no-block（重启失败不影响“成功语义”）
    """

    def __init__(
        self,
        shm_reader: ShmReader,
        conf_writer: ConfShmWriter,
        indicators: Optional[IndicatorController],
        model_path: str,
        restart_unit_name: str,
        default_sensitivity: int,
        poll_ms: int,
        max_alarm: int,
        msk_eps: int,
        print_shm: bool,
        print_roi: bool,
        print_thr: bool,
        print_msk: bool,
    ):
        # adapters
        self.reader = shm_reader
        self._confw = conf_writer
        self._gpio = indicators

        # upg
        self.model_path = str(model_path)
        self.restart_unit = str(restart_unit_name).strip()

        # params (与你原代码一致的风格：做 clamp)
        self.poll_ms = max(1, int(poll_ms))
        self.max_alarm = max(1, min(10, int(max_alarm)))
        self.msk_eps = int(max(0, int(msk_eps)))

        self.print_shm = bool(print_shm)
        self.print_roi = bool(print_roi)
        self.print_thr = bool(print_thr)
        self.print_msk = bool(print_msk)

        # shared states
        self._lock = threading.Lock()
        self._rois_rect: List[Box] = []
        self._rois_poly: List[Polygon] = []
        self._msk_rect: List[Box] = []
        self._latest_dets: List[Det] = []

        # threshold
        self._sensitivity = 50
        self._conf_thres = 0.50
        self._set_sensitivity(default_sensitivity, announce=False)

        # alarm hold
        self._alarm_active = False
        self._alarm_hold_until = 0.0

        # online stats (给 tcp_server 打印/print_shm)
        self._stat_lock = threading.Lock()
        self._online = 0
        self._cid = 0

        # UPG mutex
        self._upg_lock = threading.Lock()

        # loop control
        self._stop = False
        self._th = threading.Thread(target=self._shm_loop, daemon=True)
        self._th.start()

        self.update_indicators()

    # ---------- hooks for tcp_server ----------

    @property
    def stopped(self) -> bool:
        return bool(getattr(self, "_stop", False))

    def on_client_connected(self):
        if not hasattr(self, "_stat_lock"):
            self._stat_lock = threading.Lock()
            self._cid = 0
            self._online = 0
        with self._stat_lock:
            self._cid += 1
            self._online += 1
            return self._cid, self._online

    def on_client_disconnected(self) -> int:
        if not hasattr(self, "_stat_lock"):
            return 0
        with self._stat_lock:
            self._online = max(0, self._online - 1)
            return self._online

    # ---------- APIs for handlers ----------

    def set_rect_rois(self, rois: List[Rect]) -> None:
        # Num=0：清空 ROI 并清报警（与你原版一致）
        if not rois:
            with self._lock:
                self._rois_rect = []
                self._alarm_active = False
                self._alarm_hold_until = 0.0
            self.update_indicators()
            if self.print_roi:
                print("[TCP] SET rect_rois=[]", flush=True)
            return

        now = time.monotonic()
        with self._lock:
            self._rois_rect = list(rois)
            self._alarm_active = self._calc_alarm_active_locked()
            if self._alarm_active:
                self._alarm_hold_until = now + ALARM_HOLD_SEC
        self.update_indicators()
        if self.print_roi:
            print(f"[TCP] SET rect_rois={rois}", flush=True)

    def set_poly_rois(self, polys: List[Polygon]) -> None:
        now = time.monotonic()
        with self._lock:
            self._rois_poly = list(polys)
            self._alarm_active = self._calc_alarm_active_locked()
            if self._alarm_active:
                self._alarm_hold_until = now + ALARM_HOLD_SEC
        self.update_indicators()
        if self.print_roi:
            print(f"[TCP] POL polys(num={len(polys)})={polys}", flush=True)

    def set_masks(self, rects: List[Rect]) -> None:
        now = time.monotonic()
        with self._lock:
            self._msk_rect = list(rects)
            self._alarm_active = self._calc_alarm_active_locked()
            if self._alarm_active:
                self._alarm_hold_until = now + ALARM_HOLD_SEC
        self.update_indicators()
        if self.print_msk:
            if rects:
                print(f"[TCP] MSK rects(num={len(rects)})={rects}", flush=True)
            else:
                print("[TCP] MSK rects=[]", flush=True)

    def set_sensitivity(self, sens: int) -> None:
        self._set_sensitivity(sens, announce=True)

        now = time.monotonic()
        with self._lock:
            self._alarm_active = self._calc_alarm_active_locked()
            if self._alarm_active:
                self._alarm_hold_until = now + ALARM_HOLD_SEC
        self.update_indicators()

    def get_alarm_boxes(self) -> List[Rect]:
        # effective -> 去重 -> 截断 max_alarm（与你原版 ASK 语义一致）
        alarm: List[Rect] = []
        seen = set()

        with self._lock:
            dets = list(self._latest_dets)

        for det in dets:
            with self._lock:
                ok = self._det_is_effective_locked(det)
            if not ok:
                continue

            x1, y1, x2, y2, _sc, _cl = det
            box = (x1, y1, x2, y2)
            if box in seen:
                continue
            seen.add(box)
            alarm.append(box)
            if len(alarm) >= self.max_alarm:
                break

        return alarm

    def upgrade_model(self, data: bytes, max_model_bytes: int) -> None:
        with self._upg_lock:
            n = len(data)
            if n <= 0 or n > int(max_model_bytes):
                raise ValueError(f"UPG length invalid: {n} (max={max_model_bytes})")

            self._atomic_replace_file(self.model_path, data)

            def _restart_bg():
                try:
                    self._restart_service()
                    print(f"[UPG] restart done: {self.restart_unit}", flush=True)
                except Exception as e:
                    print(f"[UPG][ERR] restart failed: {e}", flush=True)

            threading.Thread(target=_restart_bg, daemon=True).start()

    # ---------- internal logic (match your source) ----------

    def _set_sensitivity(self, sens: int, announce: bool = True) -> None:
        sens = int(max(0, min(100, int(sens))))
        conf = sens / 100.0  # 与你原版一致
        with self._lock:
            self._sensitivity = sens
            self._conf_thres = conf
        self._confw.write_conf(conf)
        if announce and self.print_thr:
            print(f"[THR] sensitivity={sens} -> conf={conf:.2f}", flush=True)

    @staticmethod
    def _score_u8_to_f(sc: int) -> float:
        return float(int(sc)) / 255.0

    def _box_hits_any_roi_locked(self, box: Box) -> bool:
        for r in self._rois_rect:
            if aabb_intersects(box, r):
                return True
        for poly in self._rois_poly:
            if rect_poly_intersects(box, poly):
                return True
        return False

    def _box_fully_in_any_msk_locked(self, box: Box) -> bool:
        eps = int(self.msk_eps)
        for m in self._msk_rect:
            if rect_contains(m, box, eps=eps):
                return True
        return False

    def _det_is_effective_locked(self, det: Det) -> bool:
        x1, y1, x2, y2, sc, _cl = det
        if self._score_u8_to_f(sc) < float(self._conf_thres):
            return False
        box = (x1, y1, x2, y2)

        # MSK 优先：完全落在 MSK -> 永远忽略
        if self._box_fully_in_any_msk_locked(box):
            return False

        # 然后看 ROI 命中
        return self._box_hits_any_roi_locked(box)

    def _calc_alarm_active_locked(self) -> bool:
        # 无 ROI 或无 dets：直接 False（与你原版一致）
        if not (self._rois_rect or self._rois_poly) or not self._latest_dets:
            return False
        for det in self._latest_dets:
            if self._det_is_effective_locked(det):
                return True
        return False

    def _alarm_output_now(self) -> bool:
        now = time.monotonic()
        with self._lock:
            return bool(self._alarm_active) or (now < float(self._alarm_hold_until))

    def update_indicators(self) -> None:
        if self._gpio is None:
            return
        if self._alarm_output_now():
            self._gpio.set_mode(IndicatorController.MODE_ERR)
        else:
            self._gpio.set_mode(IndicatorController.MODE_OK)

    def stop(self) -> None:
        if self._stop:
            return
        self._stop = True
        try:
            if self._th.is_alive():
                self._th.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._gpio is not None:
                self._gpio.force_mode(IndicatorController.MODE_OK)
        except Exception:
            pass
        
    def online_now(self) -> int:
        with self._stat_lock:
            return self._online

    def _shm_loop(self) -> None:
        while not self._stop:
            dets = self.reader.try_read()
            if dets is not None:
                now = time.monotonic()
                with self._lock:
                    self._latest_dets = dets
                    self._alarm_active = self._calc_alarm_active_locked()
                    if self._alarm_active:
                        self._alarm_hold_until = now + ALARM_HOLD_SEC

                if self.print_shm:
                    with self._lock:
                        conf = self._conf_thres
                        eps = self.msk_eps
                        msk_n = len(self._msk_rect)
                    print(
                        f"[SHM] n={len(dets)} conf={conf:.2f} msk_n={msk_n} msk_eps={eps}px (online={self.online_now()})",
                        flush=True,
                    )

                self.update_indicators()

            time.sleep(self.poll_ms / 1000.0)

    # ---------- UPG helpers ----------

    @staticmethod
    def _fsync_dir(dir_path: str) -> None:
        try:
            dfd = os.open(dir_path, os.O_DIRECTORY | os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass

    def _atomic_replace_file(self, dst_path: str, data: bytes) -> None:
        dst_path = os.path.abspath(dst_path)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        tmp_path = os.path.join(dst_dir, f".model.rknn.tmp.{os.getpid()}.{int(time.time()*1000)}")
        bak_path = os.path.join(dst_dir, f"model.rknn.bak.{ts}")

        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # 备份旧模型（失败不影响主流程）
        if os.path.exists(dst_path):
            try:
                import shutil
                shutil.copy2(dst_path, bak_path)
            except Exception:
                pass

        os.replace(tmp_path, dst_path)
        self._fsync_dir(dst_dir)

    def _restart_service(self) -> None:
        unit = self.restart_unit
        if not unit:
            return

        cmd = ["systemctl", "restart", "--no-block", unit]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"systemctl restart {unit} failed: {r.stderr.strip() or r.stdout.strip()}")