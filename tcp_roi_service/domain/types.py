# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple

# 你的源代码里用的是 Box/Det/Point/Polygon
Box = Tuple[int, int, int, int]                 # (x1,y1,x2,y2)
Det = Tuple[int, int, int, int, int, int]       # (x1,y1,x2,y2,score_u8,cls_u8)
Point = Tuple[int, int]
Polygon = List[Point]

# 为了你新结构里叫 Rect 更直观：Rect = Box
Rect = Box