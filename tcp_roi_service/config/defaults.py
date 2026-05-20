# -*- coding: utf-8 -*-
from __future__ import annotations

# ===== GPIO (来自你的源文件) =====
BEEP_GPIO = 115  # GPIO3_C3
LEDG_GPIO = 139  # GPIO4_B3
LEDR_GPIO = 113  # GPIO3_C1
LEDB_GPIO = 108  # GPIO3_B4

# 防闪烁/报警保持（来自你的源文件）
LED_HOLD_SEC = 1.0
ALARM_HOLD_SEC = 1.0

# ===== SHM =====
DEFAULT_CONF_SHM = "/dev/shm/yolo_conf_thr"
DEFAULT_SHM_PATH = "/dev/shm/yolo_person_boxes"

# SHM v2 layout（来自你的源文件）
MAX_BOXES = 10
DET_BYTES = 12
SHM_SIZE = 1 + 2 + MAX_BOXES * DET_BYTES  # 123

# ===== UPG =====
DEFAULT_MODEL_PATH = "/home/radxa/Security_monitoring/model.rknn"
DEFAULT_RESTART_UNIT = "yolo-to-shm.service"
MAX_MODEL_BYTES = 256 * 1024 * 1024  # 256MB

# ===== TCP 默认参数（来自你的 argparse） =====
DEFAULT_BIND = "0.0.0.0"
DEFAULT_PORT = 9000
DEFAULT_CLIENT_TIMEOUT = 300

# ===== 业务默认参数（来自你的 argparse） =====
DEFAULT_POLL_MS = 20
DEFAULT_MAX_ALARM = 10
DEFAULT_DEFAULT_SENSITIVITY = 40
DEFAULT_MSK_EPS = 2

# print flags（来自你的 argparse 默认）
DEFAULT_PRINT_SHM = False
DEFAULT_PRINT_ROI = False
DEFAULT_PRINT_THR = False
DEFAULT_PRINT_MSK = True