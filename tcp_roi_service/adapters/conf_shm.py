# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import mmap
import struct

"""
conf_shm：写入 float32 little-endian 到 /dev/shm/yolo_conf_thr
保持源代码语义：conf clamp 到 [0,1]，直接写 mm[0:4]
"""
class ConfShmWriter:
    def __init__(self, path: str, size: int = 4):
        self.path = str(path)
        self.size = int(size)

        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, self.size)
        self.mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)

    def write_conf(self, conf: float) -> None:
        conf = 0.0 if conf < 0.0 else (1.0 if conf > 1.0 else float(conf))
        self.mm[0:4] = struct.pack("<f", float(conf))

    def close(self) -> None:
        try:
            self.mm.close()
        except Exception:
            pass