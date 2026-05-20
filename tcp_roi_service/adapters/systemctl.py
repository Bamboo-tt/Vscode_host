# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess

"""
systemctl 重启封装：
- 与源代码一致：systemctl restart --no-block <unit>
- 若返回码非 0：抛异常（让上层打印，但不影响 FINISH 语义）
"""


def restart_service(unit: str) -> None:
    unit = str(unit).strip()
    if not unit:
        return

    cmd = ["systemctl", "restart", "--no-block", unit]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"systemctl restart {unit} failed: {msg}")