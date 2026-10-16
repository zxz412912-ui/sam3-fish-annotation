from PIL import Image
import torch
import cv2
import numpy as np
import os

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# =====================================
# 模型初始化
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)

# =====================================
# 修改这里：换成你的测试图片路径
# =====================================

# 换成你自己的测试图片路径（图片勿上传，注意知识产权）
image_path = r"path/to/your/test_image.jpg"

image    = Image.open(image_path).convert("RGB")
image_np = np.array(image)
IMG_H, IMG_W = image_np.shape[:2]
print(f"图像尺寸: {IMG_W} x {IMG_H}")

os.makedirs("test_results", exist_ok=True)

# =====================================
# 测试用的所有 prompt
# 按部位分组，每组测试多个候选 prompt
# =====================================

test_cases = {

    # ---------- 胸鳍 ----------
    "pectoral_fin": [
        "pectoral fin",
        "pectoral fin of fish",
        "fish pectoral fin",
        "fin behind gill cover",
        "fan shaped fin on fish side",
        "fin attached to fish body side",
        "fish side fin near head",
    ],

    # ---------- 腹鳍 ----------
    "pelvic_fin": [
        "pelvic fin",
        "pelvic fin of fish",
        "fish pelvic fin",
        "small fin on fish belly",
        "ventral fin of fish",
        "fish belly fin",
        "paired fin under fish body",
    ],

    # ---------- 臀鳍 ----------
    "anal_fin": [
        "anal fin",
        "anal fin of fish",
        "fish anal fin",
        "fin under fish body near tail",
        "unpaired ventral fin near tail",
        "fish bottom rear fin",
    ],
}

# =====================================
# 工具函数
# =====================================

def get_mask_stats(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0, 0
    return int(mask.sum()), int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def save_result(image_np, mask, label, save_path):
    """把 mask 叠加到原图上保存"""
    overlay = image_np.copy()
    overlay[mask] = (0, 255, 0)
    result = cv2.addWeighted(image_np, 0.4, overlay, 0.6, 0)

    # 画轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

    # 写标签
    area, xmin, xmax, ymin, ymax = get_mask_stats(mask)
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    cv2.putText(result, label[:30], (max(cx-100, 0), cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


# =====================================
# 执行测试
# =====================================

print("\n" + "="*60)
print("开始 Prompt 测试")
print("="*60)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

    state = processor.set_image(image)

    for part_name, prompts in test_cases.items():
        print(f"\n{'='*60}")
        print(f"测试部位: {part_name}")
        print(f"{'='*60}")

        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            scores = output["scores"]
            masks  = output["masks"]

            if len(scores) == 0:
                print(f"  ✗ [{prompt}]  → 无候选")
                continue

            # 取 score 最高的 mask
            best_idx   = int(scores.argmax())
            best_score = float(scores[best_idx])
            best_mask  = masks[best_idx]
            if best_mask.ndim == 3:
                best_mask = best_mask[0]
            best_mask = best_mask.detach().cpu().numpy() > 0.5

            area, xmin, xmax, ymin, ymax = get_mask_stats(best_mask)
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            # 保存可视化
            safe_prompt = prompt.replace(" ", "_").replace("/", "-")
            save_path   = f"test_results/{part_name}__{safe_prompt}.png"
            save_result(image_np, best_mask, prompt, save_path)

            print(f"  ✓ [{prompt}]")
            print(f"      score={best_score:.4f}  area={area}"
                  f"  center=({cx},{cy})  bbox=({xmin},{xmax},{ymin},{ymax})")
            print(f"      保存: {save_path}")

print("\n" + "="*60)
print("测试完成！请查看 test_results/ 文件夹中的图片")
print("对每张图判断：位置正确？形状合理？")
print("="*60)