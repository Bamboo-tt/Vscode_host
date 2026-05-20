# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import mmap
import struct
from typing import List, Optional

from domain.types import Det

"""
SHM reader v2（123B）：
- [0]    ready: 1B
- [1:3]  length: u16 little-endian
- [3:]   dets: 10 * 12B
         x1,y1,x2,y2 (u16 LE *4) + score(u8) + cls(u8) + pad(u16)

对齐源代码的“5 秒权限重试 + chmod/fchmod + umask=0”逻辑：
- 常见问题：旧 shm 文件权限不对 / umask 导致 open 失败
- 策略：持续重试到 deadline，最终给出可读提示
"""


MAX_BOXES = 10
DET_BYTES = 12
SHM_SIZE = 1 + 2 + MAX_BOXES * DET_BYTES


class ShmReader:
    def __init__(self, path: str):
        self.path = str(path)
        existed = os.path.exists(self.path)
        deadline = time.time() + 5.0

        while True:
            old_umask = os.umask(0)
            fd = None
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)

                # 尽力把权限改到 666（源代码优先 fchmod，失败再 chmod）
                try:
                    os.fchmod(fd, 0o666)
                except Exception:
                    try:
                        os.chmod(self.path, 0o666)
                    except Exception:
                        pass

                os.ftruncate(fd, SHM_SIZE)
                self.mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

                # 新建时清零（避免脏数据）
                if not existed:
                    self.mm[:] = b"\x00" * SHM_SIZE
                return

            except PermissionError as e:
                if time.time() >= deadline:
                    raise PermissionError(
                        f"打开 SHM 失败（{self.path}），通常是权限/umask 导致。\n"
                        f"建议：rm 掉旧 shm 后再启动 Producer/Consumer。\n"
                        f"原始错误：{e}"
                    )
                time.sleep(0.2)

            finally:
                os.umask(old_umask)
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass

    def try_read(self) -> Optional[List[Det]]:
        # ready!=1：认为没有新数据
        if self.mm[0] != 1:
            return None

        n = struct.unpack("<H", self.mm[1:3])[0]
        n = max(0, min(MAX_BOXES, n))

        dets: List[Det] = []
        off = 3
        for _ in range(n):
            x1, y1, x2, y2 = struct.unpack("<HHHH", self.mm[off : off + 8])
            sc = int(self.mm[off + 8])
            cl = int(self.mm[off + 9])
            dets.append((int(x1), int(y1), int(x2), int(y2), sc, cl))
            off += DET_BYTES

        # clear ready
        self.mm[0:1] = b"\x00"
        return dets

    def close(self) -> None:
        try:
            self.mm.close()
        except Exception:
            pass