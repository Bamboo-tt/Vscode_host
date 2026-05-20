# -*- coding: utf-8 -*-
from __future__ import annotations

import struct
from app.tcp_server import recv_exact

"""
协议小工具（只做“通用协议动作”）：
- READY / FINISH / FAIL 固定回复
- 常用字段读取：u8、u16/u32（网络字节序 BE）
注意：
- TCP payload 的数字字段按你原协议使用 big-endian（network order）
- SHM 那边是 little-endian，不在这里处理
"""

READY = b"READY"
FINISH = b"FINISH"
FAIL = b"FAIL"


def send_ready(conn) -> None:
    conn.sendall(READY)


def send_finish(conn) -> None:
    conn.sendall(FINISH)


def send_fail(conn) -> None:
    conn.sendall(FAIL)


def recv_u8(conn) -> int:
    """读 1 字节无符号整数"""
    return recv_exact(conn, 1)[0]


def recv_u16_be(conn) -> int:
    """读 u16 (big-endian)"""
    return struct.unpack("!H", recv_exact(conn, 2))[0]


def recv_u32_be(conn) -> int:
    """读 u32 (big-endian)"""
    return struct.unpack("!I", recv_exact(conn, 4))[0
]