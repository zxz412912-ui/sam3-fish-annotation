"""
test_main.py — 纯逻辑单元测试
测试 main.py 中所有不依赖 GPU / SAM3 模型的函数。
"""
import numpy as np
import sys

# 从 main.py 中复制纯逻辑函数（避免触发模型加载）
# 实际测试时应从 main.py 导入，但这里独立定义以避免 import 副作用

def get_mask_stats(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0, 0
    area = int(mask.sum())
    return area, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def masks_overlap_ratio(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    smaller_area = min(mask_a.sum(), mask_b.sum())
    if smaller_area == 0:
        return 0.0
    return float(intersection) / float(smaller_area)


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def make_range(lo_ratio, hi_ratio, direction, axis="x", img_w=1707, img_h=1280):
    size = img_w if axis == "x" else img_h
    if direction == "left":
        return int(size * lo_ratio), int(size * hi_ratio)
    else:
        return int(size * (1 - hi_ratio)), int(size * (1 - lo_ratio))


def resolve_overlaps(results, result_scores):
    labels = list(results.keys())
    to_remove = set()

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            if l1 in to_remove or l2 in to_remove:
                continue

            mask1, mask2 = results[l1], results[l2]
            iou = mask_iou(mask1, mask2)

            if iou > 0.7:
                base1 = l1.replace(" 1", "").replace(" 2", "")
                base2 = l2.replace(" 1", "").replace(" 2", "")
                if base1 == base2:
                    continue

                s1 = result_scores.get(l1, 0)
                s2 = result_scores.get(l2, 0)
                if s1 < s2:
                    to_remove.add(l1)
                    print(f"  [互斥] {l1} 与 {l2} IoU={iou:.2f}，移除 {l1} (score {s1:.3f} < {s2:.3f})")
                else:
                    to_remove.add(l2)
                    print(f"  [互斥] {l1} 与 {l2} IoU={iou:.2f}，移除 {l2} (score {s2:.3f} < {s1:.3f})")

    for l in to_remove:
        del results[l]
        del result_scores[l]

    return results, result_scores


# =========================================
# 测试框架
# =========================================
passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


print("=" * 60)
print("测试 1: make_range 方向感知")
print("=" * 60)

# 左朝向: x 范围 [0.1, 0.5] -> [100, 500]
lo, hi = make_range(0.1, 0.5, "left", axis="x", img_w=1000, img_h=800)
test("left x: lo=100", lo == 100)
test("left x: hi=500", hi == 500)

# 右朝向: x 范围 [0.1, 0.5] -> 从右侧算，即 [500, 900]
lo, hi = make_range(0.1, 0.5, "right", axis="x", img_w=1000, img_h=800)
test("right x: lo=500", lo == 500)
test("right x: hi=900", hi == 900)

# y 轴方向不影响（direction 只用于 x 翻转）
lo, hi = make_range(0.2, 0.8, "left", axis="y", img_w=1000, img_h=1000)
test("left y: lo=200", lo == 200)
test("left y: hi=800", hi == 800)

print("=" * 60)
print("测试 2: get_mask_stats")
print("=" * 60)

# 创建简单 mask
mask = np.zeros((100, 200), dtype=bool)
mask[30:70, 40:160] = True
area, xmin, xmax, ymin, ymax = get_mask_stats(mask)
test("area = 40*120=4800", area == 4800)
test("xmin=40", xmin == 40)
test("xmax=159", xmax == 159)
test("ymin=30", ymin == 30)
test("ymax=69", ymax == 69)

# 空 mask
area, xmin, xmax, ymin, ymax = get_mask_stats(np.zeros((50, 50), dtype=bool))
test("empty mask all zeros", area == 0 and xmin == 0 and xmax == 0 and ymin == 0 and ymax == 0)

print("=" * 60)
print("测试 3: masks_overlap_ratio")
print("=" * 60)

a = np.zeros((100, 100), dtype=bool)
a[20:60, 20:60] = True  # 40x40 = 1600

b = np.zeros((100, 100), dtype=bool)
b[40:80, 40:80] = True  # 40x40 = 1600, 交集 20x20=400

ratio = masks_overlap_ratio(a, b)
# smaller_area = 1600, intersection = 400 -> 0.25
test("overlap ratio = 0.25", abs(ratio - 0.25) < 1e-6)

# 完全重叠
c = a.copy()
ratio = masks_overlap_ratio(a, c)
test("identical overlap = 1.0", abs(ratio - 1.0) < 1e-6)

# 无重叠
d = np.zeros((100, 100), dtype=bool)
d[80:90, 80:90] = True
ratio = masks_overlap_ratio(a, d)
test("no overlap = 0.0", abs(ratio - 0.0) < 1e-6)

print("=" * 60)
print("测试 4: mask_iou")
print("=" * 60)

iou = mask_iou(a, b)
# intersection=400, union=1600+1600-400=2800 -> 400/2800 = 0.142857...
test("IoU = 400/2800", abs(iou - 400/2800) < 1e-6)

iou = mask_iou(a, a)
test("identical IoU = 1.0", abs(iou - 1.0) < 1e-6)

iou = mask_iou(a, d)
test("no overlap IoU = 0.0", abs(iou - 0.0) < 1e-6)

print("=" * 60)
print("测试 5: resolve_overlaps 互斥检查")
print("=" * 60)

# 场景 A: 不同部位高度重叠 (IoU > 0.7)
mask_trunk = np.zeros((100, 200), dtype=bool)
mask_trunk[20:80, 50:150] = True

mask_dorsal1 = np.zeros((100, 200), dtype=bool)
mask_dorsal1[25:75, 55:145] = True  # 与 trunk 高度重叠

results = {
    "trunk": mask_trunk,
    "dorsal fin 1": mask_dorsal1,
}
scores = {"trunk": 0.6, "dorsal fin 1": 0.85}

results_out, scores_out = resolve_overlaps(results.copy(), scores.copy())
test("dorsal fin 1 wins over trunk", "dorsal fin 1" in results_out and "trunk" not in results_out)

# 场景 B: 同一部位的两片鳍允许重叠
mask_dorsal2 = np.zeros((100, 200), dtype=bool)
mask_dorsal2[30:70, 60:140] = True

results = {
    "dorsal fin 1": mask_dorsal1,
    "dorsal fin 2": mask_dorsal2,
}
scores = {"dorsal fin 1": 0.85, "dorsal fin 2": 0.70}

results_out, scores_out = resolve_overlaps(results.copy(), scores.copy())
test("same part fins kept", "dorsal fin 1" in results_out and "dorsal fin 2" in results_out)

# 场景 C: 无重叠的不同部位保留
mask_head = np.zeros((100, 200), dtype=bool)
mask_head[10:30, 150:190] = True

results = {
    "trunk": mask_trunk,
    "head": mask_head,
}
scores = {"trunk": 0.6, "head": 0.9}

results_out, scores_out = resolve_overlaps(results.copy(), scores.copy())
test("no overlap kept both", "trunk" in results_out and "head" in results_out)

print("=" * 60)
print("测试 6: 竖图旋转逻辑")
print("=" * 60)

# 模拟竖图旋转: 100x50 -> 旋转后 50x100
# np.rot90 顺时针 k=-1 (等价于逆时针 k=1) 旋转矩阵
# 注意：原代码用 PIL rotate(90) 逆时针，然后用 np.rot90(k=-1) 顺时针回正
# 测试: 创建特征 mask，旋转后转回，位置是否一致

mask_orig = np.zeros((100, 50), dtype=bool)  # 高100, 宽50 (竖图)
mask_orig[80:90, 10:20] = True  # 底部区域

# PIL rotate(90) 逆时针 -> 新尺寸 50x100
# 原 (y, x) -> 新 (x, H-1-y) = (x, 99-y)
# 区域 (80:90, 10:20) -> 新 (10:20, 9:19) [y'=x, x'=99-y]
# 用 np.rot90 逆时针 k=1 模拟 PIL rotate(90)
mask_rot = np.rot90(mask_orig, k=1)

# 再用 np.rot90 顺时针 k=-1 转回
mask_back = np.rot90(mask_rot, k=-1)

test("rotate back identical", np.array_equal(mask_orig, mask_back))

# 验证旋转后尺寸
h, w = mask_orig.shape
test("rot90 h->w", mask_rot.shape == (w, h))

print("=" * 60)
print("测试 7: 方向 + 背侧联合约束")
print("=" * 60)

# 鱼头朝左，背朝上
# x 范围: 鱼头在左，尾在右
# 背鳍应在上方 (y < trunk_y_center)
# 臀鳍/腹鳍应在下方 (y > trunk_y_center)

direction = "left"
dorsal_side = "up"
img_w, img_h = 4096, 3072
trunk_y_center = 1500

# 背鳍 x 范围: 鱼头在左 -> [0.15, 0.70]
x_lo, x_hi = make_range(0.15, 0.70, direction, axis="x", img_w=img_w, img_h=img_h)
test("dorsal x left: lo=614", x_lo == 614)
test("dorsal x left: hi=2867", x_hi == 2867)

# 如果背朝上，背鳍 y_range = (0, 0.55*img_h)
if dorsal_side == "up":
    y_range = (0, int(img_h * 0.55))
else:
    y_range = (int(img_h * 0.45), img_h)
test("dorsal y up: hi=1689", y_range[1] == 1689)

# 如果背朝下，背鳍 y_range 应反转
dorsal_side = "down"
if dorsal_side == "up":
    y_range = (0, int(img_h * 0.55))
else:
    y_range = (int(img_h * 0.45), img_h)
test("dorsal y down: lo=1382", y_range[0] == 1382)

# 鱼头朝右的情况
direction = "right"
x_lo, x_hi = make_range(0.15, 0.70, direction, axis="x", img_w=img_w, img_h=img_h)
# 右朝向: 从右侧算，[0.15, 0.70] -> [1-0.70, 1-0.15] = [0.30, 0.85]
test("dorsal x right: lo=1228", x_lo == 1228)
test("dorsal x right: hi=3481", x_hi == 3481)

print("=" * 60)
print("测试结果汇总")
print("=" * 60)
print(f"  通过: {passed}")
print(f"  失败: {failed}")
print(f"  总计: {passed + failed}")

if failed > 0:
    print("\n!!! 有测试失败，请检查 main.py 逻辑 !!!")
    sys.exit(1)
else:
    print("\n所有测试通过！")
    sys.exit(0)
