# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List
from domain.types import Box, Point, Polygon


def aabb_intersects(a: Box, b: Box) -> bool:
    """矩形 AABB 相交判定（严格面积相交），与源代码一致。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return (x2 > x1) and (y2 > y1)


def point_in_poly(p: Point, poly: Polygon) -> bool:
    """ray casting"""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def _ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def rect_poly_intersects(r: Box, poly: Polygon) -> bool:
    """矩形与多边形是否相交/包含（与你源代码同语义）"""
    rx1, ry1, rx2, ry2 = r

    px = [p[0] for p in poly]
    py = [p[1] for p in poly]
    poly_bb = (min(px), min(py), max(px), max(py))
    if not aabb_intersects(r, poly_bb):
        return False

    # poly 点落在 rect
    for x, y in poly:
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            return True

    # rect 角点落在 poly
    corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
    for c in corners:
        if point_in_poly(c, poly):
            return True

    # 边相交
    rect_edges = [
        ((rx1, ry1), (rx2, ry1)),
        ((rx2, ry1), (rx2, ry2)),
        ((rx2, ry2), (rx1, ry2)),
        ((rx1, ry2), (rx1, ry1)),
    ]
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        for c, d in rect_edges:
            if segments_intersect(a, b, c, d):
                return True

    return False


def rect_contains(outer: Box, inner: Box, eps: int = 0) -> bool:
    """outer 是否包含 inner，允许 eps 容差（MSK 用）"""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (ox1 - eps) <= ix1 and (oy1 - eps) <= iy1 and (ox2 + eps) >= ix2 and (oy2 + eps) >= iy2