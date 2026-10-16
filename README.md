# 基于 SAM3 的鳜鱼部位自动标注工具

基于 Meta [SAM3](README_SAM3.md) 视觉模型实现的鳜鱼（*Siniperca chuatsi*）8 部位自动语义分割与标注工具。输入鱼体侧面照片，自动检测并输出各部位的 LabelMe 格式标注，可配合 LabelMe 软件直接打开进行人工修正。

## 功能特点

- **8 个部位自动检测**：躯干、头部、眼睛、背鳍、臀鳍、胸鳍、腹鳍、尾鳍
- **对称鳍分离标注**：臀鳍、腹鳍若重叠但可分辨，自动拆分为两片独立标注
- **旋转不变处理**：竖向照片自动旋转处理，检测完成后自动转回原始坐标系
- **鱼体局部坐标系**：基于鱼体主轴（头→尾）建立局部坐标系，空间约束不受鱼朝向影响
- **背景干扰排除**：全局鱼体检测作为 ROI，排除盆中其他鱼的干扰
- **批量处理**：支持整个文件夹批量标注，自动跳过已处理图片
- **LabelMe 格式输出**：直接生成 LabelMe JSON，可用 LabelMe 打开继续编辑

## 环境要求

- NVIDIA GPU（推荐，使用 CUDA + bfloat16 加速）
- Python 3.10+
- SAM3 模型权重 `sam3.pt`（约 3.3 GB，需单独下载，不包含在本仓库中）

## 安装

```bash
git clone https://github.com/zxz412912-ui/sam3-fish-annotation.git
cd sam3-fish-annotation
pip install -e .
```

## 使用方法

编辑 `main.py` 末尾的配置区：

```python
# 模式 1：单张图片
image_path = r"path/to/your/image.jpg"
results = annotate_fish(image_path, output_dir="./fish_annotations")

# 模式 2：批量处理整个目录
image_dir = r"path/to/your/image_dir"      # 图片目录
image_ext = (".jpg", ".jpeg", ".png", ".bmp")
output_dir = "./fish_annotations"           # 输出目录
skip_existing = True                        # 跳过已处理图片
```

运行：

```bash
python main.py
```

## 输出说明

```
fish_annotations/
├── 图片名.json               # LabelMe 格式标注（多边形顶点）
├── 图片名.jpg/png            # 原始图片副本
├── output/
│   └── 图片名_annotated.png  # 叠加可视化效果图
└── annotations/
    └── 图片名.json           # 程序元数据（面积、bbox 等）
```

### 用 LabelMe 打开标注

```bash
labelme fish_annotations
```

或打开 LabelMe → File → Open Dir → 选择 `fish_annotations` 文件夹。

### 部位标签对照

| 中文 | 英文标签 |
|------|----------|
| 躯干 | trunk |
| 头部 | head |
| 眼睛 | eyes |
| 背鳍 | dorsal fin |
| 臀鳍 | anal fin |
| 胸鳍 | pectoral fin |
| 腹鳍 | pelvic fin |
| 尾鳍 | caudal fin |

## 项目结构

```
├── main.py              # 主程序：批量处理入口 + 标注流程编排
├── annotate_core.py     # 核心：各部位检测算法、局部坐标系、输出合成
├── test.py              # Prompt 调试工具：测试不同文本提示词效果
├── rebuild_labelme.py   # LabelMe JSON 重建工具
├── sam3/                # SAM3 模型源码（来自 Meta）
├── config.json          # SAM3 模型配置
└── processor_config.json
```

## 算法要点

1. **全局鱼体检测**：先用 "whole fish" 提示词锁定主体鱼，作为后续所有部位检测的 ROI
2. **基准部位检测**：躯干 → 尾鳍 → 头部 → 眼睛
3. **局部坐标系建立**：以躯干质心为原点、头尾连线为主轴 u、背腹方向为 v，所有空间约束都在该坐标系下表达
4. **条件放宽级联**：背鳍等难检测部位采用"严格 → 放宽 → 最宽松"三级条件逐级尝试
5. **后处理修正**：trunk 去除鳍部区域、尾鳍曲线截断 + 傅里叶轮廓平滑、尾鳍-躯干间隙填充

## 声明

- 本项目基于 Meta 的 [SAM3](https://github.com/facebookresearch/sam3) 开发，模型相关代码与配置遵循原项目许可证
- 仓库不包含模型权重与任何鱼类图像数据；测试图片请使用自有或已获授权的数据
- 自动标注结果建议经人工复核后再用于科研或生产
- 打开main.py 修改路径到你要划分部位的图片路径，即可使用
