#!/usr/bin/env python3
"""
rebuild_labelme.py — 从 labelme JSON 重建标注结果

用法：
    python rebuild_labelme.py --json <path/to/xxx.json>
    python rebuild_labelme.py --json <path/to/xxx.json> --output-dir ./new_output
    python rebuild_labelme.py --dir <path/to/annotations_dir>  # 批量重建
"""

import argparse
import cv2
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_core import build_output, get_mask_stats, get_body_axes


def polygons_to_mask(shapes, img_h, img_w):
    """
    将 labelme JSON 中的 polygon 列表转换为 mask 字典。
    同名标签的多边形会合并到同一个 mask 中。
    """
    masks = {}
    for shape in shapes:
        label = shape["label"]
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) < 3:
            continue
        if label not in masks:
            masks[label] = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(masks[label], [pts], 255)

    # 转换为 bool mask
    return {k: v > 0 for k, v in masks.items()}


def rebuild_from_labelme(json_path, output_dir=None):
    """
    从 labelme JSON 重建标注结果。
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"错误: JSON 文件不存在: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    image_path = data["imagePath"]
    shapes = data["shapes"]
    metadata = data.get("metadata", {})

    # 尝试找到对应的原始图片
    # 优先在同目录下查找，其次用 JSON 文件名推断
    img_dir = json_path.parent
    candidates = [
        img_dir / image_path,
        img_dir / image_path.replace(".json", ".jpg"),
        img_dir / image_path.replace(".json", ".png"),
        img_dir / json_path.stem + ".jpg",
        img_dir / json_path.stem + ".png",
    ]
    image_file = None
    for cand in candidates:
        if cand.exists():
            image_file = cand
            break

    if image_file is None:
        print(f"警告: 未找到原始图片，尝试用 imagePath 直接加载: {image_path}")
        # 如果 imagePath 是绝对路径或相对路径
        if os.path.exists(image_path):
            image_file = Path(image_path)
        else:
            print(f"错误: 无法找到原始图片，跳过: {json_path}")
            return None

    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    # 如果 JSON 中的尺寸和实际图片不一致，以实际图片为准
    if img_h != orig_h or img_w != orig_w:
        print(f"  [尺寸修正] JSON: {img_w}x{img_h}, 实际: {orig_w}x{orig_h}")
        img_h, img_w = orig_h, orig_w

    # 重建 mask
    results = polygons_to_mask(shapes, img_h, img_w)

    # 如果没有 trunk/head/caudal，尝试计算 body axes
    origin, u_axis, v_axis = None, None, None
    if all(k in results for k in ["trunk", "head", "caudal fin"]):
        origin, u_axis, v_axis = get_body_axes(
            results["trunk"], results["head"], results["caudal fin"]
        )
        print(f"  [body axes] 已重新计算")
    elif "trunk" in results:
        origin, u_axis, v_axis = get_body_axes(
            results["trunk"], results.get("head"), results.get("caudal fin")
        )

    # 从 metadata 读取元信息
    direction = metadata.get("direction", "left")
    dorsal_side = metadata.get("dorsal_side", "up")
    was_rotated = metadata.get("was_rotated", False)

    base_name = json_path.stem
    if output_dir is None:
        output_dir = json_path.parent

    print(f"\n{'='*60}")
    print(f"重建: {base_name}")
    print(f"图片: {image_file}")
    print(f"部位数: {len(results)}")
    print(f"{'='*60}")

    # 调用 build_output 合成最终输出
    build_output(
        results, str(image_file), image_np, output_dir, base_name,
        direction, dorsal_side, was_rotated,
        origin=origin, u_axis=u_axis, v_axis=v_axis
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="从 labelme JSON 重建标注结果")
    parser.add_argument("--json", type=str, help="单个 labelme JSON 文件路径")
    parser.add_argument("--dir", type=str, help="包含多个 labelme JSON 的目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认同 JSON 目录）")
    args = parser.parse_args()

    if args.json:
        rebuild_from_labelme(args.json, output_dir=args.output_dir)
    elif args.dir:
        json_dir = Path(args.dir)
        json_files = sorted(json_dir.glob("*.json"))
        # 排除 metadata 子目录里的 JSON
        json_files = [f for f in json_files if f.parent.name != "annotations"]
        if not json_files:
            print(f"错误: 目录 {args.dir} 下未找到 JSON 文件")
            return
        print(f"\n批量重建: {len(json_files)} 个文件")
        for jf in json_files:
            try:
                rebuild_from_labelme(jf, output_dir=args.output_dir)
            except Exception as e:
                print(f"!!! 重建失败: {jf.name} — {e}")
                import traceback
                traceback.print_exc()
    else:
        print("请提供 --json 或 --dir 参数")
        parser.print_help()


if __name__ == "__main__":
    main()
