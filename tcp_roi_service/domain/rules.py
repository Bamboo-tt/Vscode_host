# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Set

from domain.types import Box, Det, Polygon
from domain.geometry import aabb_intersects, rect_poly_intersects, rect_contains


def score_u8_to_f(sc: int) -> float:
    return float(int(sc)) / 255.0


def box_hits_any_roi(box: Box, rois_rect: List[Box], rois_poly: List[Polygon]) -> bool:
    for r in rois_rect:
        if aabb_intersects(box, r):
            return True
    for poly in rois_poly:
        if rect_poly_intersects(box, poly):
            return True
    return False


def box_fully_in_any_msk(box: Box, msk_rect: List[Box], msk_eps: int) -> bool:
    eps = int(msk_eps)
    for m in msk_rect:
        if rect_contains(m, box, eps=eps):
            return True
    return False


def det_is_effective(
    det: Det,
    conf_thres: float,
    rois_rect: List[Box],
    rois_poly: List[Polygon],
    msk_rect: List[Box],
    msk_eps: int,
) -> bool:
    """
    与你源代码 _det_is_effective_locked 等价：
    1) score_u8/255 >= conf_thres
    2) MSK 优先：完全落在 MSK -> False
    3) 命中任意 ROI（rect 或 poly）-> True，否则 False
    """
    x1, y1, x2, y2, sc, _cl = det
    if score_u8_to_f(sc) < float(conf_thres):
        return False

    box = (x1, y1, x2, y2)

    if box_fully_in_any_msk(box, msk_rect, msk_eps):
        return False

    return box_hits_any_roi(box, rois_rect, rois_poly)


def calc_alarm_active(
    dets: List[Det],
    conf_thres: float,
    rois_rect: List[Box],
    rois_poly: List[Polygon],
    msk_rect: List[Box],
    msk_eps: int,
) -> bool:
    """
    与你源代码 _calc_alarm_active_locked 等价：
    - 没 ROI 或没 dets：直接 False
    - 有任意有效 det：True
    """
    if not (rois_rect or rois_poly) or not dets:
        return False

    for det in dets:
        if det_is_effective(det, conf_thres, rois_rect, rois_poly, msk_rect, msk_eps):
            return True
    return False


def calc_alarm_boxes(
    dets: List[Det],
    conf_thres: float,
    rois_rect: List[Box],
    rois_poly: List[Polygon],
    msk_rect: List[Box],
    msk_eps: int,
    max_alarm: int,
) -> List[Box]:
    """
    与你源代码 handle_ask 输出逻辑等价：
    - 遍历 dets
    - effective 才收集
    - 去重
    - 截断 max_alarm
    """
    out: List[Box] = []
    seen: Set[Box] = set()

    for det in dets:
        if not det_is_effective(det, conf_thres, rois_rect, rois_poly, msk_rect, msk_eps):
            continue
        x1, y1, x2, y2, _sc, _cl = det
        box = (x1, y1, x2, y2)
        if box in seen:
            continue
        seen.add(box)
        out.append(box)
        if len(out) >= int(max_alarm):
            break

    return out