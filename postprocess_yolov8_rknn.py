#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_yolov8_rknn.py

作用：
- 适配 RKNN 上常见的 YOLOv8 “多输出 + DFL”结构：
  每个尺度输出：
    reg: [1, 64, H, W]   (4 * reg_max=16 的分布回归)
    cls: [1, C,  H, W]   (类别分数，可能是 logits 或已 sigmoid)
    sum: [1, 1,  H, W]   (类别分数和的分支，可忽略或用于辅助；默认不参与乘法)
- 输出：boxes(xyxy, 输入尺度), classes, scores

注意：
- 默认不把 sum 分支乘进最终 score（避免“乘小导致全过不了阈值”）。
- 若你的 cls 已经是 0~1 概率，会自动识别并跳过 sigmoid。
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _maybe_prob(x: np.ndarray) -> np.ndarray:
    """
    作用：
    - 如果数值已在 [0,1]（允许少量浮动），认为是概率
    - 否则按 logits 做 sigmoid
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if -1e-3 <= x_min and x_max <= 1.0 + 1e-3:
        return x.astype(np.float32)
    return _sigmoid(x.astype(np.float32))


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """
    作用：标准 NMS（class-agnostic）
    boxes: (N,4) in xyxy
    scores:(N,)
    return: keep indices
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int32)

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    areas = w * h

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= float(iou_thres))[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def _to_nchw(x: np.ndarray, num_classes: int) -> np.ndarray:
    """
    作用：尽量把输出统一成 NCHW
    支持：
    - NCHW: [1,C,H,W]
    - NHWC: [1,H,W,C]
    - CHW : [C,H,W] -> [1,C,H,W]
    - HWC : [H,W,C] -> [1,C,H,W]
    """
    x = np.array(x)

    if x.ndim == 4:
        # 可能是 NCHW: [N,C,H,W] 或 NHWC: [N,H,W,C]
        n, a, b, c = x.shape

        cand_c = (1, 64, int(num_classes))

        # 更可靠的判断：看最后一维是否像“通道数”
        if c in cand_c and a not in cand_c:
            return np.transpose(x, (0, 3, 1, 2))
        if a in cand_c:
            return x

        # 兜底：谁更像通道谁就是 C（通道一般 <=256）
        if c <= 256 and a > 256:
            return np.transpose(x, (0, 3, 1, 2))
        return x

    if x.ndim == 3:
        a, b, c = x.shape
        cand_c = (1, 64, int(num_classes))

        # CHW
        if a in cand_c:
            return x[None, ...]
        # HWC
        if c in cand_c:
            return np.transpose(x, (2, 0, 1))[None, ...]

        # 兜底：通道一般更小
        if c <= 256 and a > 256:
            return np.transpose(x, (2, 0, 1))[None, ...]
        return x[None, ...]

    raise ValueError(f"Unsupported output ndim: {x.shape}")


def _group_branches(outputs, num_classes: int):
    """
    作用：把 outputs 按 feature map 尺度分组，得到若干 branch：
      branch = dict(reg=..., cls=..., aux=...)
    """
    fmap = {}  # key=(H,W) -> dict

    for o in outputs:
        t = _to_nchw(o, num_classes=num_classes)
        _, ch, h, w = t.shape
        key = (h, w)
        if key not in fmap:
            fmap[key] = {}

        if ch == 64:
            fmap[key]["reg"] = t
        elif ch == num_classes:
            # 单类时 ch==1 与 aux 分支会冲突：优先占用为 cls
            if "cls" not in fmap[key]:
                fmap[key]["cls"] = t
            else:
                fmap[key]["aux"] = t
        elif ch == 1:
            # 可能是 sum/aux 分支（默认不参与 score 计算）
            if "aux" not in fmap[key]:
                fmap[key]["aux"] = t
        else:
            # 其他形状忽略
            pass

    branches = []
    for (h, w), d in fmap.items():
        if "reg" in d and "cls" in d:
            branches.append(((h, w), d["reg"], d["cls"], d.get("aux", None)))

    # 按 feature map 从大到小（80x80 -> 40x40 -> 20x20）
    branches.sort(key=lambda x: x[0][0] * x[0][1], reverse=True)
    return branches


def yolov8_rknn_post_process(
    outputs,
    input_w: int,
    input_h: int,
    num_classes: int = 1,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    topk: int = 300,
    person_only: bool = False,
    person_id: int = 0,
    class_aware_nms: bool = False,
):
    """
    返回：
      boxes:  (M,4) float32，xyxy，坐标在输入尺度（0~input_w/input_h）
      classes:(M,)  int32
      scores: (M,)  float32
    """
    outs = [np.array(o) for o in outputs]

    # 单输出（某些导出链路会这样），这里不展开：你当前模型主要是多输出
    if len(outs) == 1:
        # 直接返回空，避免误解码（需要的话你再告诉我你的单输出 shape）
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    branches = _group_branches(outs, num_classes=num_classes)
    if not branches:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    reg_max = 16  # YOLOv8 默认 DFL bins
    acc = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max)

    all_boxes = []
    all_scores = []
    all_classes = []

    for (hm, wm), reg, cls, aux in branches:
        reg = reg.astype(np.float32)  # [1,64,H,W]
        cls = cls.astype(np.float32)  # [1,C,H,W]  (C=num_classes)

        stride_x = float(input_w) / float(wm)
        stride_y = float(input_h) / float(hm)

        if person_only:
            # 只取 person 通道（更快）
            cls_map = _maybe_prob(cls[0, int(person_id)])  # [H,W]
            score_map = cls_map
            cls_id_map = np.full((hm, wm), int(person_id), dtype=np.int32)
        else:
            cls_prob = _maybe_prob(cls[0])  # [C,H,W]
            cls_id_map = np.argmax(cls_prob, axis=0).astype(np.int32)  # [H,W]
            score_map = np.max(cls_prob, axis=0).astype(np.float32)    # [H,W]

        flat = score_map.reshape(-1)
        keep0 = np.where(flat >= float(conf_thres))[0]
        if keep0.size == 0:
            continue

        k = int(min(int(topk), int(keep0.size)))
        cand = flat[keep0]
        idx_local = np.argpartition(cand, -k)[-k:]
        idx_local = idx_local[np.argsort(cand[idx_local])[::-1]]
        idx = keep0[idx_local]

        ys = (idx // wm).astype(np.int32)
        xs = (idx % wm).astype(np.int32)
        scores = flat[idx].astype(np.float32)
        classes = cls_id_map.reshape(-1)[idx].astype(np.int32)

        # DFL decode：只对候选点取 reg
        reg_hw = reg[0].transpose(1, 2, 0)               # [H,W,64]
        reg_vec = reg_hw[ys, xs, :].reshape(-1, 4, reg_max)  # [N,4,16]
        p = _softmax(reg_vec, axis=2)                    # [N,4,16]
        dist = (p * acc).sum(axis=2)                     # [N,4] l,t,r,b

        cx = (xs.astype(np.float32) + 0.5)
        cy = (ys.astype(np.float32) + 0.5)

        x1 = (cx - dist[:, 0]) * stride_x
        y1 = (cy - dist[:, 1]) * stride_y
        x2 = (cx + dist[:, 2]) * stride_x
        y2 = (cy + dist[:, 3]) * stride_y

        b = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(input_w))
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(input_h))

        all_boxes.append(b)
        all_scores.append(scores)
        all_classes.append(classes)

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)

    # 全局 topk（避免 NMS 太慢）
    if scores.size > int(topk):
        ii = np.argsort(-scores)[: int(topk)]
        boxes = boxes[ii]
        scores = scores[ii]
        classes = classes[ii]

    # NMS
    if person_only or (not class_aware_nms):
        keep = nms_xyxy(boxes, scores, float(iou_thres))
        return boxes[keep], classes[keep], scores[keep]

    # class-aware NMS
    keep_all = []
    for c in np.unique(classes):
        m = (classes == c)
        if not np.any(m):
            continue
        b_c = boxes[m]
        s_c = scores[m]
        keep_c = nms_xyxy(b_c, s_c, float(iou_thres))
        idx_c = np.where(m)[0][keep_c]
        keep_all.append(idx_c)

    if not keep_all:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32)

    keep = np.concatenate(keep_all)
    # 再按分数排序一次
    keep = keep[np.argsort(-scores[keep])]
    return boxes[keep], classes[keep], scores[keep]
