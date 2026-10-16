from PIL import Image, ImageOps
import torch
import cv2
import numpy as np
import json
import os
from pathlib import Path

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("加载SAM3模型...")

# =====================================
# 模型初始化
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

# =====================================
# 工具函数
# =====================================


from annotate_core import *

def annotate_fish(image_path, output_dir="./fish_annotations"):
    """对单张图片进行鳜鱼部位标注。"""

    original_image = Image.open(image_path).convert("RGB")
    original_image = ImageOps.exif_transpose(original_image)
    orig_w, orig_h = original_image.size

    # 判断是否为竖图（高 > 宽 * 1.2）
    is_vertical = orig_h > orig_w * 1.2
    was_rotated = False

    if is_vertical:
        print(f"  [竖图] 逆时针旋转90°处理")
        image = original_image.rotate(90, expand=True)
        was_rotated = True
    else:
        image = original_image

    image_np = np.array(image)
    IMG_H, IMG_W = image_np.shape[:2]

    print(f"\n{'='*60}")
    print(f"图像: {image_path}")
    print(f"原始尺寸: {orig_w} x {orig_h}")
    if was_rotated:
        print(f"旋转后尺寸: {IMG_W} x {IMG_H}")
    print(f"{'='*60}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(image_path).stem

    results = {}
    result_scores = {}

    with torch.autocast(device_type=device, dtype=torch.bfloat16):

        state = processor.set_image(image)

        # 0. 全局鱼体检测（排除背景中的其他鱼）
        print("--- 全局鱼体 ---")
        whole_mask = detect_whole_fish(processor, state, IMG_W, IMG_H)

        # 1. 躯干
        print("--- 躯干 ---")
        mask, score = detect_normal_part(processor, state, part_prompts["trunk"], "躯干", whole_mask=whole_mask)
        if mask is not None:
            results["trunk"] = mask
            result_scores["trunk"] = score

        # 2. 尾鳍（单部位，只保留最佳候选）
        print("--- 尾鳍 ---")
        tail_candidates = collect_candidates(processor, state, part_prompts["caudal fin"], "尾鳍")
        # 过滤面积
        area_lo = int(IMG_W * IMG_H * 0.0002)
        area_hi = int(IMG_W * IMG_H * 0.030)
        valid = [c for c in tail_candidates if area_lo <= c["area"] <= area_hi]
        if valid:
            # 优先选择远离头部的候选（尾部方向）
            if "head" in results:
                _, hxmin, hxmax, _, _ = get_mask_stats(results["head"])
                head_cx = (hxmin + hxmax) / 2
                best = max(valid, key=lambda c: abs(c["cx"] - head_cx))
            else:
                best = max(valid, key=lambda c: c["score"])
            results["caudal fin"] = best["mask"]
            result_scores["caudal fin"] = best["score"]
            print(f"  尾鳍: score={best['score']:.4f} area={best['area']} x=({best['xmin']},{best['xmax']}) y=({best['ymin']},{best['ymax']})")
        else:
            print("  未找到尾鳍")

        # 3. 头部（用于朝向判断 + 胸鳍定位）
        print("--- 头部 ---")
        mask, score = detect_normal_part(processor, state, part_prompts["head"], "头部", whole_mask=whole_mask)
        head_mask = mask
        if mask is not None:
            results["head"] = mask
            result_scores["head"] = score

        # 4. 眼睛
        print("--- 眼睛 ---")
        mask, score = detect_normal_part(processor, state, part_prompts["eyes"], "眼睛", whole_mask=whole_mask)
        if mask is not None:
            results["eyes"] = mask
            result_scores["eyes"] = score

        # ---- 建立鱼体局部坐标系（旋转不变）----
        print("\n--- 建立鱼体局部坐标系 ---")
        origin, u_axis, v_axis = get_body_axes(
            results.get("trunk"), results.get("head"), results.get("caudal fin")
        )
        if origin is None:
            print("  [警告] 无法建立局部坐标系，fallback 到全局坐标")
            origin = np.array([IMG_W / 2, IMG_H / 2])
            u_axis = np.array([1.0, 0.0])
            v_axis = np.array([0.0, -1.0])

        # 不再用眼睛判断背侧，直接依赖初始 v_axis（基于鱼体主轴）
        # 背侧 v 大，腹侧 v 小，通过收紧范围确保分隔
        dorsal_side = "up"

        u_min, u_max, v_min, v_max = get_trunk_extents(results.get("trunk"), origin, u_axis, v_axis)
        v_span = v_max - v_min
        print(f"  局部坐标系: u∈[{u_min:.1f}, {u_max:.1f}], v∈[{v_min:.1f}, {v_max:.1f}]")

        # ---- 收集各部位候选并计算局部坐标（用于后续检测）----
        dorsal_cands = collect_candidates(processor, state, part_prompts["dorsal fin"], "", whole_mask=whole_mask)
        anal_cands = collect_candidates(processor, state, part_prompts["anal fin"], "", whole_mask=whole_mask)
        pelvic_cands = collect_candidates(processor, state, part_prompts["pelvic fin"], "", whole_mask=whole_mask)
        to_local_candidates(dorsal_cands, origin, u_axis, v_axis)
        to_local_candidates(anal_cands, origin, u_axis, v_axis)
        to_local_candidates(pelvic_cands, origin, u_axis, v_axis)

        # ---- 朝向判断（保留用于 JSON 输出）----
        print("\n--- 鱼朝向判断 ---")
        direction = detect_fish_direction(results.get("head"), results.get("caudal fin"), IMG_W)
        print(f"  → 鱼头朝向: {direction}")

        # 如果尾鳍没找到，按局部坐标重新搜索（尾部 u 偏大）
        if "caudal fin" not in results:
            print("--- 尾鳍（按局部坐标重新搜索） ---")
            tail_candidates = collect_candidates(processor, state, part_prompts["caudal fin"], "尾鳍")
            to_local_candidates(tail_candidates, origin, u_axis, v_axis)
            area_lo = int(IMG_W * IMG_H * 0.0002)
            area_hi = int(IMG_W * IMG_H * 0.030)
            u_tail_lo = u_min + (u_max - u_min) * 0.3
            valid = [c for c in tail_candidates if c["local_u"] >= u_tail_lo and area_lo <= c["area"] <= area_hi]
            if valid:
                best = max(valid, key=lambda c: c["score"])
                results["caudal fin"] = best["mask"]
                result_scores["caudal fin"] = best["score"]
                print(f"  尾鳍: score={best['score']:.4f} local_u={best.get('local_u', 0):.1f}")
            else:
                print("  未找到尾鳍")

        # 6. 背鳍（单部位，仅在明显可见时标注）
        print("--- 背鳍 ---")
        if dorsal_cands:
            trunk_mask = results.get("trunk")
            u_span = u_max - u_min  # 计算 u 范围跨度

            def try_dorsal(score_thresh, v_max_allowed, overlap_thresh, u_max_allowed, label):
                dorsal_valid = []
                for c in dorsal_cands:
                    if c["score"] < score_thresh:
                        continue
                    if c["local_v"] > v_max_allowed:
                        continue
                    # 质心位置检查
                    if c["local_u"] > u_max_allowed:
                        continue
                    # mask 像素范围检查：背鳍不应延伸到臀鳍/尾部区域
                    ys_m, xs_m = np.where(c["mask"])
                    if len(xs_m) > 0:
                        points = np.column_stack([xs_m, ys_m]).astype(np.float32)
                        local_us = np.dot(points - origin, u_axis)
                        if float(local_us.max()) > u_max_allowed:
                            continue
                    if trunk_mask is not None:
                        overlap = masks_overlap_ratio(trunk_mask, c["mask"])
                        if overlap > overlap_thresh:
                            continue
                    dorsal_valid.append(c)
                if dorsal_valid:
                    best = max(dorsal_valid, key=lambda c: c["score"])
                    results["dorsal fin"] = best["mask"]
                    result_scores["dorsal fin"] = best["score"]
                    best_overlap = masks_overlap_ratio(trunk_mask, best["mask"]) if trunk_mask is not None else 0.0
                    print(f"  背鳍 [{label}]: score={best['score']:.4f} local_v={best['local_v']:.1f} overlap={best_overlap:.2f} → 保留")
                    return True
                return False

            # 第一轮：严格条件（背侧，中前部，中等重叠）
            if not try_dorsal(0.3, 0, 0.85, u_min + u_span * 0.50, "严格"):
                # 第二轮：放宽条件（靠近中心线，后部稍延，更多重叠）
                if not try_dorsal(0.2, v_max * 0.3, 0.9, u_min + u_span * 0.60, "放宽"):
                    # 第三轮：最宽松（几乎不限制）
                    if not try_dorsal(0.15, v_max * 0.5, 0.95, u_min + u_span * 0.70, "最宽松"):
                        print("  背鳍未检测到或条件不满足 → 跳过")
        else:
            print("  背鳍未检测到 → 跳过")

        # 7. 臀鳍（尾部，上下两个）
        print("--- 臀鳍 ---")
        u_span = u_max - u_min
        u_tail_lo = u_min + u_span * 0.70   # 严格限制在尾部
        u_tail_hi = u_max
        # 臀鳍在尾部上下都有，不限制 v 方向
        m1, m2, state, s1, s2 = detect_symmetric_fins(
            processor, state, part_prompts["anal fin"], "anal fin",
            u_range=(u_tail_lo, u_tail_hi),
            v_range=(v_min, v_max),   # 不限制 v，允许上下两个
            avoid_masks=[head_mask, results.get("dorsal fin"), results.get("caudal fin")],  # 避免与背鳍、尾鳍重叠
            overlap_thresh=0.25,
            v_diff_thresh=max(v_span * 0.02, 5),  # 降低阈值，更容易检测上下两片
            whole_mask=whole_mask,
            origin=origin, u_axis=u_axis, v_axis=v_axis,
            v_split=True
        )
        if m1 is not None:
            results["anal fin 1"] = m1
            result_scores["anal fin 1"] = s1
        if m2 is not None:
            results["anal fin 2"] = m2
            result_scores["anal fin 2"] = s2

        # 8. 胸鳍（文本 + 几何 bbox 双重 prompt）
        mask1, _, state, s1, _ = detect_pectoral_fin(
            processor, image, state, part_prompts["pectoral fin"], head_mask,
            origin, u_axis, v_axis, u_min, u_max, v_min, v_max
        )
        if mask1 is not None:
            results["pectoral fin"] = mask1
            result_scores["pectoral fin"] = s1

        # 9. 腹鳍（躯干中部，胸鳍和臀鳍之间，鱼的下侧）
        print("--- 腹鳍 ---")
        # 腹鳍在躯干中部，胸鳍后方，臀鳍前方
        u_pelvic_lo = u_min + u_span * 0.25
        u_pelvic_hi = u_min + u_span * 0.55
        v_pelvic_range = (v_span * 0.05, v_max)  # 腹侧
        m1, m2, state, s1, s2 = detect_symmetric_fins(
            processor, state, part_prompts["pelvic fin"], "pelvic fin",
            u_range=(u_pelvic_lo, u_pelvic_hi),
            v_range=v_pelvic_range,
            avoid_masks=[head_mask],
            overlap_thresh=0.25,
            v_diff_thresh=max(v_span * 0.04, 10),
            whole_mask=whole_mask,
            origin=origin, u_axis=u_axis, v_axis=v_axis
        )
        if m1 is not None:
            results["pelvic fin 1"] = m1
            result_scores["pelvic fin 1"] = s1
        if m2 is not None:
            results["pelvic fin 2"] = m2
            result_scores["pelvic fin 2"] = s2

    # 部位互斥检查
    print("\n--- 部位互斥检查 ---")
    results, result_scores = resolve_overlaps(results, result_scores)

    # 如果旋转过，把结果旋转回原始坐标系
    if was_rotated:
        print("\n[旋转回原始坐标系]")
        for label in list(results.keys()):
            # 顺时针旋转90°回原始坐标系
            results[label] = np.rot90(results[label], k=-1)
        # 使用原始图像进行可视化
        image_np = np.array(original_image)
        IMG_H, IMG_W = image_np.shape[:2]
    else:
        image_np = np.array(image)

    # =====================================================================

    # 合成输出 + 保存 labelme JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    build_output(results, image_path, image_np, output_dir, base_name,
                 direction, dorsal_side, was_rotated,
                 origin=origin, u_axis=u_axis, v_axis=v_axis)

    # 复制原始图片到输出目录，方便 labelme 打开目录
    import shutil
    dst_img = Path(output_dir) / Path(image_path).name
    if not dst_img.exists():
        shutil.copy2(image_path, dst_img)
        print(f"原始图片已复制至: {dst_img}")

    save_labelme_json(results, image_path, image_np, output_dir, base_name,
                      direction, dorsal_side, was_rotated)

    return results
if __name__ == "__main__":
    # ============================================================
    # 使用方式：修改下面的参数即可
    # ============================================================

    # 模式 1：单张图片
    # image_path = r"path/to/your/test_image.jpg"
    # results = annotate_fish(image_path, output_dir="./fish_annotations")

    # 模式 2：批量处理整个目录
    image_dir = r"D:\鱼\images鳜鱼"
    image_ext = (".jpg", ".jpeg", ".png", ".bmp")  # 支持的图片后缀
    output_dir = "./fish_annotations"
    skip_existing = True  # 为 True 时跳过已处理过的图片（根据结果文件判断）

    # 收集所有图片
    all_images = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(image_ext):
            all_images.append(os.path.join(image_dir, f))
    all_images.sort()

    total = len(all_images)
    if total == 0:
        print(f"错误: 目录 {image_dir} 下没有 {image_ext} 格式的图片")
        exit(1)

    print(f"\n{'='*60}")
    print(f"批量处理开始")
    print(f"目录: {image_dir}")
    print(f"图片数量: {total}")
    print(f"输出目录: {output_dir}")
    print(f"跳过已处理: {skip_existing}")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_files = []

    for i, img_path in enumerate(all_images, 1):
        base_name = Path(img_path).stem
        # 跳过已处理的文件：检查标注图是否存在（现在在 output/ 子目录）
        if skip_existing:
            vis_path = Path(output_dir) / "output" / f"{base_name}_annotated.png"
            if vis_path.exists():
                print(f"[{i}/{total}] 跳过 (已处理): {base_name}")
                skip_count += 1
                continue

        print(f"\n{'='*60}")
        print(f"[{i}/{total}] 处理: {base_name}")
        print(f"{'='*60}")

        try:
            annotate_fish(img_path, output_dir=output_dir)
            success_count += 1
        except Exception as e:
            print(f"\n!!! 处理失败: {base_name}")
            print(f"!!! 错误: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            failed_files.append(base_name)

    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"  总计: {total}")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    if failed_files:
        print(f"  失败列表: {failed_files}")
    print(f"{'='*60}")
