from PIL import Image, ImageOps
import torch
import cv2
import numpy as np
import json
import os
from pathlib import Path

# =====================================
# 分类依据（来自 问题1.docx）
# =====================================
#
# 标注类型（8个部位）：
#   1. 躯干     → trunk
#   2. 尾鳍     → caudal fin
#   3. 头部     → head
#   4. 眼睛     → eyes
#   5. 背鳍     → dorsal fin
#   6. 臀鳍     → anal fin
#   7. 胸鳍     → pectoral fin
#   8. 腹鳍     → pelvic fin
#
# 物种：鳜鱼（Siniperca chuatsi）
#
# 标注顺序：躯干 → 尾鳍 → 头部 → 眼睛 → 背鳍 → 臀鳍 → 胸鳍 → 腹鳍
#
# 关键规则：鱼鳍是对称的，如果有重叠（但是能看出来），就分开标记，标成两个
# =====================================

part_prompts = {
    "trunk": [
        "fish body",
        "body of fish",
        "fish trunk"
    ],
    "head": [
        "fish head",
        "head of fish"
    ],
    "eyes": [
        "fish eye",
        "eye of fish"
    ],
    "dorsal fin": [
        "fish dorsal fin",
        "back fin of fish",
        "fin on fish back"
    ],
    "pectoral fin": [
        "complete pectoral fin with radiating fin rays and spotted membrane",
        "pectoral fin extending from gill cover to body side with visible rays",
        "full fan-shaped pectoral fin on lateral side of fish body near head",
        "pectoral fin with spreading fin rays and spotted texture behind gill cover",
        "pectoral fin with visible branching fin rays and transparent membrane"
    ],
    "anal fin": [
        "fish anal fin",
        "fish rear bottom fin",
        "rear ventral fin"
    ],
    "pelvic fin": [
        "fish pelvic fin",
        "fish belly fin",
        "small ventral fin"
    ],
    "caudal fin": [
        "fish tail fin",
        "caudal fin",
        "tail fin of fish"
    ],
    "whole fish": [
        "whole fish",
        "complete fish",
        "entire fish"
    ]
}

# =====================================
# 工具函数
# =====================================

def get_mask_stats(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0, 0
    area = int(mask.sum())
    return area, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def smooth_contour_fourier(contour, n_harmonics=12):
    """
    对闭合轮廓做傅里叶级数平滑，去除锯齿和波浪。
    n_harmonics: 保留的谐波数量，越小轮廓越平滑。
    """
    pts = contour[:, 0, :].astype(np.float32)
    n = len(pts)
    # 点数不足时直接返回
    if n < 2 * n_harmonics + 1:
        return contour
    
    # 参数化：累积弧长
    diffs = np.diff(pts, axis=0, append=pts[0:1])
    arc = np.cumsum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
    t = arc / arc[-1] if arc[-1] > 0 else np.linspace(0, 1, n)
    
    # 等间距重采样（确保周期性）
    t_uniform = np.linspace(0, 1, n)
    x_uniform = np.interp(t_uniform, t, pts[:, 0])
    y_uniform = np.interp(t_uniform, t, pts[:, 1])
    
    # FFT 平滑：去除高频成分
    x_fft = np.fft.fft(x_uniform)
    y_fft = np.fft.fft(y_uniform)
    
    # 只保留 DC + n_harmonics 个正频率 + n_harmonics 个负频率
    x_fft[n_harmonics+1:-n_harmonics] = 0
    y_fft[n_harmonics+1:-n_harmonics] = 0
    
    x_smooth = np.real(np.fft.ifft(x_fft))
    y_smooth = np.real(np.fft.ifft(y_fft))
    
    # 重新整形为 OpenCV 轮廓格式
    smoothed = np.column_stack([x_smooth, y_smooth]).reshape(-1, 1, 2).astype(np.int32)
    return smoothed


# =====================================
# 旋转不变：鱼体局部坐标系（基于 PCA/主轴）
# =====================================

def get_body_axes(trunk_mask, head_mask, caudal_mask):
    """
    基于 trunk/head/caudal 计算鱼体局部坐标系（旋转不变）。
    返回: (origin, u_axis, v_axis)
    - origin: (ox, oy) 鱼体中心（trunk 质心）
    - u_axis: 主轴单位向量，从 head 指向 caudal（头->尾）
    - v_axis: 背腹轴单位向量，垂直于 u_axis，初始方向为逆时针90度。
              后续会根据 dorsal/anal 的分布自动翻转，确保 dorsal 侧 v > 0。
    """
    if trunk_mask is None:
        return None, None, None

    ys, xs = np.where(trunk_mask)
    ox, oy = float(xs.mean()), float(ys.mean())
    origin = np.array([ox, oy])

    # 主轴：优先用 head -> caudal
    if head_mask is not None and caudal_mask is not None:
        hys, hxs = np.where(head_mask)
        cys, cxs = np.where(caudal_mask)
        hx, hy = float(hxs.mean()), float(hys.mean())
        cx, cy = float(cxs.mean()), float(cys.mean())
        vec = np.array([cx - hx, cy - hy])
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            u_axis = vec / norm
        else:
            u_axis = np.array([1.0, 0.0])
    elif head_mask is not None:
        hys, hxs = np.where(head_mask)
        hx, hy = float(hxs.mean()), float(hys.mean())
        vec = np.array([ox - hx, oy - hy])
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            u_axis = vec / norm
        else:
            u_axis = np.array([1.0, 0.0])
    elif caudal_mask is not None:
        cys, cxs = np.where(caudal_mask)
        cx, cy = float(cxs.mean()), float(cys.mean())
        vec = np.array([cx - ox, cy - oy])
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            u_axis = vec / norm
        else:
            u_axis = np.array([1.0, 0.0])
    else:
        # 只有 trunk，用 PCA
        coords = np.column_stack([xs, ys]).astype(np.float32)
        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u_axis = eigvecs[:, np.argmax(eigvals)]
        # 默认让 u 指向图像右侧（x 增加方向）
        if u_axis[0] < 0:
            u_axis = -u_axis

    # 背腹轴：垂直于主轴，逆时针90度
    v_axis = np.array([-u_axis[1], u_axis[0]])

    # 统一 v 轴正方向始终指向图像下方（腹侧），确保无论鱼头朝左还是朝右：
    # v > 0  → 图像下方（腹侧）
    # v < 0  → 图像上方（背侧）
    if v_axis[1] < 0:
        v_axis = -v_axis

    return origin, u_axis, v_axis


def to_local(point, origin, u_axis, v_axis):
    """将全局坐标点 (x, y) 转换为局部坐标 (u, v)"""
    p = np.array(point) - origin
    u = float(np.dot(p, u_axis))
    v = float(np.dot(p, v_axis))
    return u, v


def to_local_candidates(candidates, origin, u_axis, v_axis):
    """给候选列表添加局部坐标字段 local_u, local_v"""
    for c in candidates:
        c["local_u"], c["local_v"] = to_local((c["cx"], c["cy"]), origin, u_axis, v_axis)
    return candidates


def get_trunk_extents(trunk_mask, origin, u_axis, v_axis):
    """获取 trunk 在局部坐标系中的 u_min, u_max, v_min, v_max"""
    if trunk_mask is None or origin is None:
        return -1.0, 1.0, -1.0, 1.0
    ys, xs = np.where(trunk_mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    local = np.dot(points - origin, np.column_stack([u_axis, v_axis]))
    return float(local[:, 0].min()), float(local[:, 0].max()), float(local[:, 1].min()), float(local[:, 1].max())


def align_v_axis_with_eyes(origin, u_axis, v_axis, head_mask, eyes_mask):
    """
    根据 eyes 相对于 head 的位置调整 v_axis，使背侧为 v > 0。
    鱼的眼睛位于身体中轴线的背侧，因此从 head 中心到 eyes 的向量
    在 v_axis 上的投影指向背侧。
    """
    if head_mask is None or eyes_mask is None or origin is None:
        return v_axis

    hys, hxs = np.where(head_mask)
    eys, exs = np.where(eyes_mask)
    if len(hxs) == 0 or len(exs) == 0:
        return v_axis

    head_center = np.array([float(hxs.mean()), float(hys.mean())])
    eye_center = np.array([float(exs.mean()), float(eys.mean())])

    # 从 head 中心到 eyes 中心的向量
    vec = eye_center - head_center

    # 投影到 v_axis
    proj_v = float(np.dot(vec, v_axis))

    if proj_v < 0:
        print(f"  [眼睛背侧] eye 投影到 v_axis = {proj_v:.1f} < 0，翻转 v_axis 使背侧 v>0")
        return -v_axis
    else:
        print(f"  [眼睛背侧] eye 投影到 v_axis = {proj_v:.1f} > 0，v_axis 方向正确")
        return v_axis


def masks_overlap_ratio(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    smaller_area  = min(mask_a.sum(), mask_b.sum())
    if smaller_area == 0:
        return 0.0
    return float(intersection) / float(smaller_area)


def mask_iou(mask_a, mask_b):
    """计算两个 mask 的 IoU（交并比）"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def estimate_dorsal_side(dorsal_cands, anal_cands, pelvic_cands, origin, u_axis, v_axis):
    """
    基于局部坐标系投票判断背侧方向。
    如果 dorsal 的平均 local_v < 0，则翻转 v_axis 使得 dorsal 位于 v > 0。
    返回 (dorsal_side, v_axis)  — dorsal_side 为 'up' 表示 v > 0 是背侧
    """
    if origin is None:
        print("  [投票] 局部坐标系缺失，默认背侧 v>0")
        return "up", v_axis

    def get_best_v(cands):
        if not cands:
            return None
        best = max(cands, key=lambda c: c["score"])
        return best.get("local_v", 0)

    v_dorsal = get_best_v(dorsal_cands)
    v_anal = get_best_v(anal_cands)
    v_pelvic = get_best_v(pelvic_cands)

    up_votes = 0
    down_votes = 0

    if v_dorsal is not None:
        if v_dorsal > 0:
            up_votes += 2
            print(f"  [投票] dorsal local_v={v_dorsal:.1f} > 0 → 背侧 v>0 +2")
        else:
            down_votes += 2
            print(f"  [投票] dorsal local_v={v_dorsal:.1f} < 0 → 背侧 v<0 +2")

    if v_anal is not None:
        if v_anal < 0:
            up_votes += 1
            print(f"  [投票] anal local_v={v_anal:.1f} < 0 → 背侧 v>0 +1")
        else:
            down_votes += 1
            print(f"  [投票] anal local_v={v_anal:.1f} > 0 → 背侧 v<0 +1")

    if v_pelvic is not None:
        if v_pelvic < 0:
            up_votes += 1
            print(f"  [投票] pelvic local_v={v_pelvic:.1f} < 0 → 背侧 v>0 +1")
        else:
            down_votes += 1
            print(f"  [投票] pelvic local_v={v_pelvic:.1f} > 0 → 背侧 v<0 +1")

    if up_votes > down_votes:
        print(f"  [结果] 背侧为 v>0（up={up_votes}, down={down_votes}）")
        return "up", v_axis
    elif down_votes > up_votes:
        print(f"  [结果] 背侧为 v<0（up={up_votes}, down={down_votes}），翻转 v_axis")
        return "down", -v_axis
    else:
        print(f"  [结果] 默认背侧 v>0（up={up_votes}, down={down_votes}）")
        return "up", v_axis


def collect_candidates(processor, state, prompts, debug_name="", whole_mask=None):
    """
    收集候选 mask，支持全局鱼体约束（whole_mask）。
    如果提供了 whole_mask，候选必须与鱼体 mask 有至少 20% 的重叠，
    以排除背景中的其他物体（如盆中的其他鱼）。
    """
    candidates = []
    for prompt in prompts:
        output = processor.set_text_prompt(state=state, prompt=prompt)
        scores = output["scores"]
        masks  = output["masks"]
        if len(scores) == 0:
            continue
        for idx in range(len(scores)):
            score = float(scores[idx])
            mask  = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.detach().cpu().numpy() > 0.5
            area, xmin, xmax, ymin, ymax = get_mask_stats(mask)
            if area < 50:
                continue
            # 新增：全局鱼体约束，排除背景干扰
            if whole_mask is not None:
                overlap_with_fish = masks_overlap_ratio(whole_mask, mask)
                if overlap_with_fish < 0.2:
                    if debug_name:
                        print(f"  [{debug_name}] prompt='{prompt}' idx={idx} "
                              f"score={score:.4f} 与鱼体重叠不足 ({overlap_with_fish:.2f})，排除")
                    continue
            if debug_name:
                print(f"  [{debug_name}] prompt='{prompt}' idx={idx} "
                      f"score={score:.4f} area={area} "
                      f"x=({xmin},{xmax}) y=({ymin},{ymax})")
            candidates.append({
                "mask": mask, "score": score, "area": area,
                "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                "cx": (xmin + xmax) // 2, "cy": (ymin + ymax) // 2,
            })
    return candidates


def make_local_range(lo_ratio, hi_ratio, lo_val, hi_val):
    """基于局部坐标系的比例范围计算"""
    span = hi_val - lo_val
    return lo_val + span * lo_ratio, lo_val + span * hi_ratio


def get_trunk_y_center(trunk_mask, img_h):
    """
    获取躯干 mask 的 y 中心线，用于判断鳍是否在腹侧。
    """
    if trunk_mask is None:
        return int(img_h * 0.5)
    _, _, _, ymin, ymax = get_mask_stats(trunk_mask)
    return int((ymin + ymax) / 2)


# =====================================
# 全局鱼体检测
# =====================================

def detect_whole_fish(processor, state, img_w, img_h):
    """
    检测完整的鱼体，作为全局 ROI，排除背景干扰（如盆中的其他鱼）。
    选择面积最大且靠近图像中心的候选。
    """
    candidates = collect_candidates(processor, state, part_prompts["whole fish"], "whole_fish")
    if not candidates:
        return None

    img_cx, img_cy = img_w / 2, img_h / 2
    best = None
    best_score = -1

    for c in candidates:
        # 面积必须足够大（至少占图像 5%）
        if c["area"] < img_w * img_h * 0.05:
            continue
        # 距离图像中心越近越好
        dist_to_center = np.sqrt((c["cx"] - img_cx) ** 2 + (c["cy"] - img_cy) ** 2)
        # 综合评分：面积越大越好，距离中心越近越好
        score = c["score"] * (1 - dist_to_center / max(img_w, img_h))
        if score > best_score:
            best_score = score
            best = c

    if best:
        print(f"  whole_fish: score={best['score']:.4f} area={best['area']}")
        return best["mask"]
    return None


# =====================================
# 方向检测
# =====================================

def detect_fish_direction(head_mask, caudal_mask, img_w):
    if head_mask is None or caudal_mask is None:
        print("  [朝向] 缺少mask，默认鱼头朝左")
        return "left"
    _, hxmin, hxmax, _, _ = get_mask_stats(head_mask)
    _, cxmin, cxmax, _, _ = get_mask_stats(caudal_mask)
    head_cx   = (hxmin + hxmax) / 2
    caudal_cx = (cxmin + cxmax) / 2
    direction = "left" if head_cx < caudal_cx else "right"
    print(f"  [朝向] head_cx={head_cx:.0f}  caudal_cx={caudal_cx:.0f}"
          f"  → 鱼头朝{'左' if direction=='left' else '右'}")
    return direction


# =====================================
# 普通部位检测
# =====================================

def detect_normal_part(processor, state, prompts, part_name="", score_thresh=0.0, whole_mask=None):
    candidates = collect_candidates(processor, state, prompts, debug_name=part_name, whole_mask=whole_mask)
    if not candidates:
        return None, -1
    best = max(candidates, key=lambda c: c["score"])
    if best["score"] < score_thresh:
        print(f"  {part_name} score={best['score']:.4f} 低于阈值 {score_thresh}，跳过")
        return None, best["score"]
    print(f"  {part_name}: score={best['score']:.4f} area={best['area']}")
    return best["mask"], best["score"]


# =====================================
# 对称鱼鳍检测（通用）
# =====================================
# 分类依据规则："鱼鳍是对称的，如果有重叠（但是能看出来），就分开标记，标成两个"

def detect_symmetric_fins(processor, state, prompts, part_name,
                          u_range=None, v_range=None,
                          avoid_masks=None,
                          overlap_thresh=0.25,
                          v_diff_thresh=None,
                          whole_mask=None,
                          origin=None, u_axis=None, v_axis=None,
                          v_split=False):
    """
    检测对称的鱼鳍（左右/上下两片）。
    关键规则：重叠但不过分（0.03 ~ 0.85）且 v 中心差异足够 → 分开标记。
    所有空间约束使用局部坐标系（u_range, v_range）。
    当 v_split=True 时，按 v 方向（背侧/腹侧）分别取最佳候选，
    确保上下两片都被检测到。

    Returns:
        (mask1, mask2, state, score1, score2)
    """
    print(f"\n===== {part_name} 检测 =====")

    candidates = collect_candidates(processor, state, prompts, debug_name=part_name, whole_mask=whole_mask)

    # 计算局部坐标
    if origin is not None and u_axis is not None and v_axis is not None:
        to_local_candidates(candidates, origin, u_axis, v_axis)

    valid = []
    for c in candidates:
        if u_range is not None and not (u_range[0] <= c.get("local_u", c["cx"]) <= u_range[1]):
            continue
        if v_range is not None and not (v_range[0] <= c.get("local_v", c["cy"]) <= v_range[1]):
            continue
        # 避免与已标注部位高度重叠
        if avoid_masks:
            skip = False
            for am in avoid_masks:
                if am is not None and masks_overlap_ratio(am, c["mask"]) > 0.85:
                    skip = True
                    break
            if skip:
                continue
        valid.append(c)

    if not valid:
        print(f"  未找到 {part_name}")
        return None, None, state, 0.0, 0.0

    # v_split 模式：按 v 方向（背侧 v<0 / 腹侧 v>=0）分别取最佳
    if v_split and origin is not None and v_axis is not None:
        up_group = [c for c in valid if c.get("local_v", c["cy"]) < 0]   # 背侧（v<0）
        down_group = [c for c in valid if c.get("local_v", c["cy"]) >= 0]  # 腹侧（v>=0）

        fin1 = max(up_group, key=lambda c: c["score"]) if up_group else None
        fin2 = max(down_group, key=lambda c: c["score"]) if down_group else None

        if fin1:
            print(f"  {part_name}_1 (背侧 v<0): score={fin1['score']:.4f} area={fin1['area']} "
                  f"x=({fin1['xmin']},{fin1['xmax']}) y=({fin1['ymin']},{fin1['ymax']}) "
                  f"local_u={fin1.get('local_u', 0):.1f} local_v={fin1.get('local_v', 0):.1f}")
        if fin2:
            print(f"  {part_name}_2 (腹侧 v>=0): score={fin2['score']:.4f} area={fin2['area']} "
                  f"x=({fin2['xmin']},{fin2['xmax']}) y=({fin2['ymin']},{fin2['ymax']}) "
                  f"local_u={fin2.get('local_u', 0):.1f} local_v={fin2.get('local_v', 0):.1f}")
        if fin1 is None and fin2 is None:
            print(f"  未找到 {part_name}")
        elif fin1 is None or fin2 is None:
            print(f"  只检测到 1 片 {part_name}")

        return (fin1["mask"] if fin1 else None), (fin2["mask"] if fin2 else None), state, \
               (fin1["score"] if fin1 else 0.0), (fin2["score"] if fin2 else 0.0)

    # 标准模式：按 score 排序，寻找第二片
    valid.sort(key=lambda c: c["score"], reverse=True)
    fin1 = valid[0]
    print(f"  {part_name}_1: score={fin1['score']:.4f} area={fin1['area']} "
          f"x=({fin1['xmin']},{fin1['xmax']}) y=({fin1['ymin']},{fin1['ymax']}) "
          f"local_u={fin1.get('local_u', 0):.1f} local_v={fin1.get('local_v', 0):.1f}")

    # 寻找第二片（对称鳍）
    fin2 = None
    if v_diff_thresh is None:
        v_diff_thresh = 10  # 默认局部坐标差异

    for c in valid[1:]:
        overlap = masks_overlap_ratio(fin1["mask"], c["mask"])
        lv1 = fin1.get("local_v", fin1["cy"])
        lv2 = c.get("local_v", c["cy"])
        # 关键规则：重叠但不过分（0.03 ~ 0.85）→ 分开标记
        if 0.03 < overlap < 0.85 and abs(lv1 - lv2) > v_diff_thresh:
            fin2 = c
            print(f"  {part_name}_2: score={fin2['score']:.4f} area={fin2['area']} "
                  f"x=({fin2['xmin']},{fin2['xmax']}) y=({fin2['ymin']},{fin2['ymax']}) "
                  f"overlap={overlap:.2f}")
            break
        # 如果完全不重叠（比如尾鳍上下两片），也分开标记
        elif overlap < 0.03 and abs(lv1 - lv2) > v_diff_thresh:
            fin2 = c
            print(f"  {part_name}_2: score={fin2['score']:.4f} area={fin2['area']} "
                  f"x=({fin2['xmin']},{fin2['xmax']}) y=({fin2['ymin']},{fin2['ymax']}) "
                  f"overlap={overlap:.2f} (无重叠)")
            break

    # 如果仍然找不到，放宽条件再试一次
    if fin2 is None:
        relaxed_thresh = max(v_diff_thresh * 0.3, 3)
        for c in valid[1:]:
            overlap = masks_overlap_ratio(fin1["mask"], c["mask"])
            lv1 = fin1.get("local_v", fin1["cy"])
            lv2 = c.get("local_v", c["cy"])
            if abs(lv1 - lv2) > relaxed_thresh and overlap < 0.95:
                fin2 = c
                print(f"  [放宽] {part_name}_2: score={fin2['score']:.4f} area={fin2['area']} "
                      f"v_diff={abs(lv1-lv2):.1f} overlap={overlap:.2f}")
                break

    if fin2 is None:
        print(f"  只检测到 1 片 {part_name}")

    return fin1["mask"], (fin2["mask"] if fin2 else None), state, fin1["score"], (fin2["score"] if fin2 else 0.0)


# =====================================
# 胸鳍检测（文本 + 几何 bbox 双重 prompt）
# =====================================

def estimate_pectoral_box(head_mask, origin, u_axis, v_axis, img_w, img_h):
    """基于局部坐标系估算胸鳍的归一化 bbox（旋转不变）"""
    if head_mask is None or origin is None:
        return [0.5, 0.5, 0.1, 0.1]

    hys, hxs = np.where(head_mask)
    head_w = int(hxs.max() - hxs.min())
    head_h = int(hys.max() - hys.min())
    
    # 从 head 后缘（远离嘴的一侧，即 +u 方向）开始
    # 对于鱼头朝左：head 后缘 = x_max（最右侧）
    # 对于鱼头朝右：head 后缘 = x_min（最左侧）
    # u_axis 从 head 指向尾，所以 head 后缘 = head 中心 + u_axis * (head_w/2)
    head_cx = float(hxs.mean())
    head_cy = float(hys.mean())
    head_rear_x = head_cx + u_axis[0] * (head_w * 0.5)
    head_rear_y = head_cy + u_axis[1] * (head_w * 0.5)
    
    # 沿 u_axis（头→尾方向）偏移约 0.15 倍 head 宽度，到达胸鳍位置
    offset = u_axis * (head_w * 0.15)
    cx = (head_rear_x + offset[0]) / img_w
    cy = (head_rear_y + offset[1]) / img_h

    w = (head_w * 0.70) / img_w
    h = (head_h * 0.55) / img_h

    cx = float(np.clip(cx, 0, 1))
    cy = float(np.clip(cy, 0, 1))
    w = float(np.clip(w, 0.02, 0.3))
    h = float(np.clip(h, 0.02, 0.3))
    print(f"  [胸鳍bbox估算] head_rear=({head_rear_x:.0f},{head_rear_y:.0f}) cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")
    return [cx, cy, w, h]


def run_geometric_pectoral(processor, image, state, head_mask, origin, u_axis, v_axis):
    results = []
    for text_prompt in ["pectoral fin on fish side",
                        "fin on lateral side of fish body",
                        "fan shaped fin attached to fish flank"]:
        processor.reset_all_prompts(state)
        state = processor.set_image(image, state)
        state = processor.set_text_prompt(state=state, prompt=text_prompt)
        box = estimate_pectoral_box(head_mask, origin, u_axis, v_axis, image.width, image.height)
        state = processor.add_geometric_prompt(box=box, label=True, state=state)

        scores = state.get("scores", [])
        masks = state.get("masks", [])
        if len(scores) == 0:
            continue
        for idx in range(len(scores)):
            score = float(scores[idx])
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.detach().cpu().numpy() > 0.5
            area, xmin, xmax, ymin, ymax = get_mask_stats(mask)
            if area < 50:
                continue
            print(f"  [胸鳍geo] prompt='{text_prompt}' idx={idx} "
                  f"score={score:.4f} area={area} "
                  f"x=({xmin},{xmax}) y=({ymin},{ymax})")
            results.append({
                "mask": mask, "score": score, "area": area,
                "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                "cx": (xmin + xmax) // 2, "cy": (ymin + ymax) // 2,
            })

    processor.reset_all_prompts(state)
    state = processor.set_image(image, state)
    return results, state


def detect_pectoral_fin(processor, image, state, prompts, head_mask, origin, u_axis, v_axis, u_min, u_max, v_min, v_max, trunk_v_center=None):
    print("\n===== 胸鳍检测 =====")

    area_lo = int((u_max - u_min) * (v_max - v_min) * 0.0003)
    area_hi = int((u_max - u_min) * (v_max - v_min) * 0.08)
    u_span = u_max - u_min
    v_span = v_max - v_min

    # 计算 head 后缘的局部 u 坐标（胸鳍应该在这里）
    head_rear_u = None
    if head_mask is not None:
        hys, hxs = np.where(head_mask)
        head_cx = float(hxs.mean())
        head_cy = float(hys.mean())
        head_w = int(hxs.max() - hxs.min())
        # head 后缘 = 质心 + u_axis * (head_w/2)
        head_rear_x = head_cx + u_axis[0] * (head_w * 0.5)
        head_rear_y = head_cy + u_axis[1] * (head_w * 0.5)
        head_rear_u, _ = to_local((head_rear_x, head_rear_y), origin, u_axis, v_axis)
        # 胸鳍在 head 后缘附近，从后缘稍微靠前到稍微靠后
        u_lo = head_rear_u - u_span * 0.05
        u_hi = head_rear_u + u_span * 0.15
        print(f"  head 后缘 local_u={head_rear_u:.1f}, 胸鳍 u 范围=[{u_lo:.1f}, {u_hi:.1f}]")
    else:
        u_lo, u_hi = make_local_range(0.15, 0.35, u_min, u_max)

    # 胸鳍在鳃盖后方侧面，覆盖较大的背腹范围
    v_lo = -v_span * 0.30   # 放宽到侧线附近
    v_hi = v_span * 0.30    # 放宽到腹侧附近

    text_candidates = collect_candidates(processor, state, prompts, debug_name="胸鳍-文本")
    geo_candidates = []
    if head_mask is not None:
        geo_candidates, state = run_geometric_pectoral(
            processor, image, state, head_mask, origin, u_axis, v_axis
        )

    # 给候选计算局部坐标
    all_candidates = text_candidates + geo_candidates
    to_local_candidates(all_candidates, origin, u_axis, v_axis)

    valid = []
    for c in all_candidates:
        if c["area"] < area_lo or c["area"] > area_hi:
            continue
        if not (u_lo <= c["local_u"] <= u_hi):
            continue
        if not (v_lo <= c["local_v"] <= v_hi):
            continue
        # 胸鳍在 head 侧面，与 head 有中等重叠是正常的，不过滤
        valid.append(c)

    if not valid:
        print("未找到胸鳍")
        return None, None, state, 0.0, 0.0

    # 选择策略：在 head 后缘附近（±0.15*u_span），选面积最大的候选
    # 这样能找到覆盖整个胸鳍的大区域，而不是只找到根部的小块
    if head_rear_u is not None:
        near_candidates = [c for c in valid if abs(c["local_u"] - head_rear_u) <= u_span * 0.15]
        if near_candidates:
            # 在附近的候选中，按面积排序，面积大的优先（覆盖整个胸鳍）
            near_candidates.sort(key=lambda c: c["area"], reverse=True)
            fin1 = near_candidates[0]
            print(f"  [选择策略] 在 head_rear_u ±{u_span*0.15:.1f} 范围内，选面积最大: {fin1['area']}")
        else:
            # 附近没有，退回到全局按score排序
            valid.sort(key=lambda c: c["score"], reverse=True)
            fin1 = valid[0]
            print(f"  [选择策略] 附近无候选，退回到 score 最高")
    else:
        valid.sort(key=lambda c: c["score"], reverse=True)
        fin1 = valid[0]
    
    # 侧视图中只标注1个胸鳍
    print(f"胸鳍: score={fin1['score']:.4f} area={fin1['area']} "
          f"x=({fin1['xmin']},{fin1['xmax']}) y=({fin1['ymin']},{fin1['ymax']}) "
          f"local_u={fin1.get('local_u', 0):.1f} local_v={fin1.get('local_v', 0):.1f}")

    return fin1["mask"], None, state, fin1["score"], 0.0


# =====================================
# 腹鳍检测
# =====================================

def detect_pelvic_fin(processor, state, prompts, origin, u_axis, v_axis, u_min, u_max, v_min, v_max, dorsal_side="up"):
    print("\n===== 腹鳍检测 =====")

    u_lo, u_hi = make_local_range(0.10, 0.55, u_min, u_max)
    area_lo = int((u_max - u_min) * (v_max - v_min) * 0.0004)
    area_hi = int((u_max - u_min) * (v_max - v_min) * 0.030)

    # 腹鳍在腹部一侧（v < 0，如果 dorsal_side="up" 即 v>0 是背侧）
    if dorsal_side == "up":
        v_range = (v_min, 0)
    else:
        v_range = (0, v_max)

    candidates = collect_candidates(processor, state, prompts, debug_name="腹鳍")
    to_local_candidates(candidates, origin, u_axis, v_axis)

    valid = []
    for c in candidates:
        if c["area"] < area_lo or c["area"] > area_hi:
            continue
        if not (u_lo <= c["local_u"] <= u_hi):
            continue
        if not (v_range[0] <= c["local_v"] <= v_range[1]):
            continue
        valid.append(c)

    if not valid:
        print("未找到腹鳍")
        return None, None, 0.0, 0.0

    valid.sort(key=lambda c: c["score"], reverse=True)
    fin1 = valid[0]
    # 侧视图中只标注1个腹鳍（可能只看到一侧）
    print(f"腹鳍: score={fin1['score']:.4f} area={fin1['area']} "
          f"x=({fin1['xmin']},{fin1['xmax']}) y=({fin1['ymin']},{fin1['ymax']}) "
          f"local_u={fin1.get('local_u', 0):.1f} local_v={fin1.get('local_v', 0):.1f}")

    return fin1["mask"], None, fin1["score"], 0.0


# =====================================
# 部位互斥检查
# =====================================

# 取消：部位不再互斥（所有部位独立存在，只画边界线）
# 如果后续需要恢复，取消下面的注释：
# def resolve_overlaps(results, result_scores):
#     ...
#     return results, result_scores

# 替代的 no-op 函数，什么都不做
def resolve_overlaps(results, result_scores):
    """所有部位不互斥，不做重叠处理。"""
    print("  [部位不互斥] 跳过重叠检查")
    return results, result_scores


# =====================================
# 主推理流程
# 标注顺序（严格遵循分类依据）：
# 躯干 → 尾鳍 → 头部 → 眼睛 → 背鳍 → 臀鳍 → 胸鳍 → 腹鳍
# =====================================


# =====================================
# 新增：build_output — 从检测结果合成最终输出
# =====================================

def build_output(results, image_path, image_np, output_dir, base_name,
                 direction, dorsal_side, was_rotated, origin=None, u_axis=None, v_axis=None):
    """从检测结果合成最终输出。
    results: dict {label: mask}，已旋转回原始坐标系
    image_np: RGB numpy array，原始图像
    """
    IMG_H, IMG_W = image_np.shape[:2]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- 重新计算 body axes（如果未提供）----
    if origin is None or u_axis is None or v_axis is None:
        if all(k in results for k in ["trunk", "head", "caudal fin"]):
            origin, u_axis, v_axis = get_body_axes(
                results.get("trunk"), results.get("head"), results.get("caudal fin")
            )

    # ---- 1. trunk 修正（去掉鱼鳍，加回头部/眼睛/胸鳍）----
    if "trunk" in results:
        print("--- 修正 trunk 范围 ---")
        trunk_mask = results["trunk"].copy()
        fins_to_remove = ["caudal fin", "dorsal fin", "anal fin 1", "anal fin 2", "pelvic fin 1", "pelvic fin 2"]
        for label in fins_to_remove:
            if label in results:
                trunk_mask = np.logical_and(trunk_mask, np.logical_not(results[label]))
                print(f"  trunk 去掉 {label}")
        for keep in ["head", "eyes", "pectoral fin"]:
            if keep in results:
                trunk_mask = np.logical_or(trunk_mask, results[keep])
                print(f"  trunk 加回 {keep}")
        results["trunk"] = trunk_mask
        area, _, _, _, _ = get_mask_stats(trunk_mask)
        print(f"  trunk 修正后面积: {area}")

    # ---- 2. 尾鳍修正（曲线截断 + 平滑 + 间隙填充）----
    if "caudal fin" in results and origin is not None and u_axis is not None and v_axis is not None:
        print("--- 修正尾鳍边界（曲线截断）---")
        tail_mask = results["caudal fin"].copy()
        ys, xs = np.where(tail_mask)
        if len(xs) > 0:
            points = np.column_stack([xs, ys]).astype(np.float32)
            local_us = np.dot(points - origin, u_axis)
            local_vs = np.dot(points - origin, v_axis)
            u_min_tail = float(local_us.min())
            u_max_tail = float(local_us.max())
            u_span_tail = u_max_tail - u_min_tail
            v_min_tail = float(local_vs.min())
            v_max_tail = float(local_vs.max())
            v_span_tail = v_max_tail - v_min_tail
            v_mid = (v_min_tail + v_max_tail) / 2.0

            base_clip = 0.03
            curve_depth = 0.02

            h, w = tail_mask.shape
            yy, xx = np.indices((h, w))
            grid_points = np.stack([xx, yy], axis=-1).astype(np.float32)
            grid_u = np.dot(grid_points.reshape(-1, 2) - origin, u_axis).reshape(h, w)
            grid_v = np.dot(grid_points.reshape(-1, 2) - origin, v_axis).reshape(h, w)

            v_norm = (grid_v - v_mid) / (v_span_tail / 2 + 1e-6)
            v_shape = 1.0 - np.clip(v_norm**2, 0, 1)
            clip_threshold = u_min_tail + u_span_tail * (base_clip + curve_depth * v_shape)

            tail_clipped = np.logical_and(tail_mask, grid_u >= clip_threshold)
            removed_pixels = np.logical_and(tail_mask, grid_u < clip_threshold)

            if "trunk" in results:
                results["trunk"] = np.logical_or(results["trunk"], removed_pixels)
                print(f"  尾鳍曲线截断后，将 {removed_pixels.sum()} 像素加回 trunk")

            tail_float = tail_clipped.astype(np.float32)
            tail_blur = cv2.GaussianBlur(tail_float, (41, 41), 10)
            tail_clipped = (tail_blur > 0.5)

            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            tail_clipped = cv2.morphologyEx(tail_clipped.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tail_clipped = cv2.morphologyEx(tail_clipped, cv2.MORPH_OPEN, kernel_open)
            tail_clipped = tail_clipped > 0

            contours, _ = cv2.findContours(tail_clipped.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                main_contour = contours[0]
                smooth_c = smooth_contour_fourier(main_contour, n_harmonics=12)
                new_mask = np.zeros_like(tail_clipped, dtype=np.uint8)
                cv2.fillPoly(new_mask, [smooth_c], 255)
                tail_clipped = new_mask > 0

            results["caudal fin"] = tail_clipped
            area, _, _, _, _ = get_mask_stats(tail_clipped)
            print(f"  尾鳍平滑后面积: {area}")

            if "trunk" in results:
                tail_only = results["caudal fin"].copy()
                trunk_only = results["trunk"].copy()

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                tail_dilated = cv2.dilate(tail_only.astype(np.uint8), kernel, iterations=2) > 0
                trunk_dilated = cv2.dilate(trunk_only.astype(np.uint8), kernel, iterations=2) > 0

                occupied = np.zeros_like(tail_only, dtype=bool)
                for m in results.values():
                    occupied = np.logical_or(occupied, m)

                gap = np.logical_and(tail_dilated, trunk_dilated)
                gap = np.logical_and(gap, np.logical_not(occupied))

                if gap.any():
                    results["trunk"] = np.logical_or(results["trunk"], gap)
                    print(f"  尾鳍-躯干间隙填充: {gap.sum()} 像素")

                union_tail_trunk = np.logical_or(results["caudal fin"], results["trunk"])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                union_closed = cv2.morphologyEx(union_tail_trunk.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                new_closed = np.logical_and(union_closed > 0, np.logical_not(union_tail_trunk))
                if new_closed.any():
                    results["trunk"] = np.logical_or(results["trunk"], new_closed)
                    print(f"  闭运算额外填充: {new_closed.sum()} 像素")

                print("  重新互斥 trunk（只去掉尾鳍）...")
                if "caudal fin" in results:
                    results["trunk"] = np.logical_and(results["trunk"], np.logical_not(results["caudal fin"]))
                area_t, _, _, _, _ = get_mask_stats(results["trunk"])
                print(f"  trunk 重新互斥后面积: {area_t}")

    # ---- 3. 生成合成标注图 ----
    print(f"\n{'='*60}")
    print("检测到的部位：")
    for k in results.keys():
        print(f"  √ {k}")
    print(f"{'='*60}")

    overlay = image_np.copy()

    colors = {
        "trunk":          (255,   0,   0),
        "head":           (  0, 255,   0),
        "eyes":           (  0,   0, 255),
        "dorsal fin":     (255, 255,   0),
        "anal fin 1":     (128, 255,   0),
        "anal fin 2":     (100, 200,   0),
        "pectoral fin":   (255, 128,   0),
        "pelvic fin":     (  0, 255, 255),
        "caudal fin":     (255,   0, 255),
    }

    draw_order = [
        "trunk", "caudal fin", "head", "eyes",
        "dorsal fin", "anal fin 1", "anal fin 2",
        "pectoral fin", "pelvic fin 1", "pelvic fin 2"
    ]

    for label in draw_order:
        if label not in results:
            continue
        color_key = "pelvic fin" if label.startswith("pelvic fin") else label
        overlay[results[label]] = colors.get(color_key, (200, 200, 200))

    result_img = cv2.addWeighted(image_np, 0.4, overlay, 0.6, 0)

    for label in draw_order:
        if label not in results:
            continue
        mask = results[label].astype(np.uint8)
        mask_smooth = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            max_area = cv2.contourArea(contours[0])
            contours = [c for c in contours if cv2.contourArea(c) >= max_area * 0.1]
        cv2.drawContours(result_img, contours, -1, (255, 255, 255), 3)
        area, xmin, xmax, ymin, ymax = get_mask_stats(results[label])
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        text_label = "pelvic fin" if label.startswith("pelvic fin") else label
        cv2.putText(result_img, text_label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    vis_path = out_path / "output" / f"{base_name}_annotated.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"标注图已保存至: {vis_path}")

    # 4. 保存 JSON 标注元信息
    json_data = {
        "image_path": str(image_path),
        "image_width": IMG_W,
        "image_height": IMG_H,
        "species": "Siniperca chuatsi",
        "direction": direction,
        "dorsal_side": dorsal_side,
        "was_rotated": was_rotated,
        "annotations": {}
    }
    for label, mask in results.items():
        area, xmin, xmax, ymin, ymax = get_mask_stats(mask)
        json_data["annotations"][label] = {
            "area": area,
            "bbox": [xmin, ymin, xmax, ymax],
        }

    json_path = out_path / "annotations" / f"{base_name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON 标注已保存至: {json_path}")

    print(f"\n{'='*60}")
    print(f"完成: {base_name}")
    print(f"{'='*60}\n")

    return results

# =====================================
# 新增：save_labelme_json — 保存 labelme 格式 JSON
# =====================================

def save_labelme_json(results, image_path, image_np, output_dir, base_name,
                      direction, dorsal_side, was_rotated):
    """将 mask 结果保存为 labelme 格式 JSON（对称部位合并为同一标签）"""
    h, w = image_np.shape[:2]

    # 标签映射：合并对称部位
    label_map = {
        "anal fin 1": "anal fin",
        "anal fin 2": "anal fin",
        "pelvic fin 1": "pelvic fin",
        "pelvic fin 2": "pelvic fin",
    }

    shapes = []
    for label, mask in results.items():
        display_label = label_map.get(label, label)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).tolist()
            shapes.append({
                "label": display_label,
                "points": pts,
                "shape_type": "polygon",
                "group_id": None,
                "description": "",
                "flags": {},
            })

    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),  # 相对路径，要求图片和 JSON 同目录
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "metadata": {
            "species": "Siniperca chuatsi",
            "direction": direction,
            "dorsal_side": dorsal_side,
            "was_rotated": was_rotated,
        }
    }

    json_path = Path(output_dir) / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Labelme JSON 已保存至: {json_path}")
    return json_path

