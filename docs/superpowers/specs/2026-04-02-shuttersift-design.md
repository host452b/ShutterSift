# ShutterSift — Design Spec

**Date:** 2026-04-02  
**Status:** Approved  
**Scope:** v0.1.0 MVP

---

## 1. Problem Statement

单反摄影师每次拍摄产出大量照片（连拍、多角度、重复构图），手动初筛耗时费力。ShutterSift 是一个本地 CLI 工具，通过多阶段 CV 管线对照片量化打分，自动分类为 Keep / Review / Reject，大幅减轻初筛体力劳动。

---

## 2. Design Principles

- **傻瓜相机理念**：`shuttersift ./photos` 一条命令，零配置开箱即用
- **本地优先**：全部核心功能离线运行，无强制网络依赖
- **单次安装，运行时自适应**：`pip install shuttersift` 安装全量依赖，运行时按硬件/模型/key 自动选最优路径，无需用户感知
- **量化透明**：每张照片产出 0–100 综合分 + 子分明细 + 文字理由
- **非破坏性**：默认只复制/创建符号链接，不移动原始文件

---

## 3. Architecture

### 3.1 分析管线（六阶段串行，早停优化）

```
输入目录
  │
  ├─ Stage 0: 扫描 & 连拍分组
  │   └─ 按 EXIF DateTimeOriginal 时间戳，间隔 ≤ 2s 归为同一 burst 组
  │       EXIF 缺失时退化：按文件名序列号相邻且创建时间差 ≤ 2s 判断
  │
  ├─ Stage 1: 加载
  │   └─ RAW (rawpy extract_thumb) → 嵌入JPEG / JPEG → numpy BGR
  │
  ├─ Stage 2: 技术质量  [纯CPU，永远运行，硬门槛]
  │   ├─ sharpness: Laplacian方差 + FFT高频能量  → [0–100]
  │   ├─ exposure:  直方图截断百分比分析          → [0–100]
  │   └─ noise:     BRISQUE (scikit-image)        → [0–100]
  │   ↑ sharpness < 30 → 直接 Reject，跳过后续所有阶段
  │
  ├─ Stage 3: 人脸分析  [MediaPipe，CPU，每张都运行]
  │   ├─ 人脸检测: MediaPipe Face Detection
  │   ├─ 检测到人脸时:
  │   │   ├─ 眼睛状态: Face Landmarker blendshapes eyeBlinkLeft/Right → [0–1]
  │   │   ├─ 笑容:     mouthSmileLeft/Right + cheekSquintLeft/Right   → [0–1]
  │   │   └─ 人脸位置: 构图辅助（主体是否偏离三分法节点）
  │   └─ 未检测到人脸时: face_quality 子分固定 75（中性），跳过闭眼硬拒绝
  │   ↑ 检测到人脸 且 所有人脸均闭眼 (eye_open < 0.25) → 直接 Reject
  │
  ├─ Stage 4: 审美 & 构图  [有GPU时用MUSIQ，否则BRISQUE替代]
  │   ├─ 审美分: pyiqa MUSIQ (GPU) / BRISQUE (CPU fallback) → [0–100]
  │   └─ 构图分: 规则引擎（人脸位置三分法 + 主体边界裁切检测）→ [0–100]
  │
  ├─ Stage 5: 去重  [pHash，纯CPU]
  │   └─ burst组内：所有帧打完分后保留最高分，其余标 Reject(duplicate)
  │
  └─ Stage 6: VLM 解释  [可选，仅 Review 档]
      ├─ 检测路径 1: ~/.shuttersift/models/*.gguf 存在
      │   └─ llama-cpp-python (Metal/CUDA/CPU 自动选)
      ├─ 检测路径 2: ANTHROPIC_API_KEY / OPENAI_API_KEY 存在
      │   └─ 调用 API (claude-haiku / gpt-4o-mini，省token)
      └─ 两者都无 → 跳过，reasons 字段留空
```

### 3.2 量化评分系统

**综合分 = 加权平均（0–100）**

| 子分 | 默认权重 | 来源 | 无人脸时处理 |
|------|---------|------|------------|
| `sharpness` | 30% | Laplacian方差标准化 | 正常计算 |
| `exposure` | 15% | 直方图分析 | 正常计算 |
| `aesthetic` | 25% | MUSIQ / BRISQUE | 正常计算 |
| `face_quality` | 20% | 眼睛×笑容×人脸清晰度 | 固定 75 分（中性） |
| `composition` | 10% | 规则引擎 | 固定 50 分 |

**决策阈值（config.yaml 可覆盖）**

```
score ≥ 70  →  Keep    
score 40–69 →  Review  
score  < 40 →  Reject  
```

**硬拒绝（不进评分）**

- `sharpness_raw < 30`（严重模糊/失焦）
- 有人脸且所有人脸 `eye_open < 0.25`

**burst 组选优**：连拍组内所有成员完成评分后，最高分者保持原决策，其余覆盖为 `Reject(duplicate)`。

### 3.3 运行时能力自适应

```python
# 伪代码：启动时探测，结果缓存
capabilities = {
    "gpu":      torch.cuda.is_available() or is_mps_available(),
    "musiq":    try_import("pyiqa"),
    "gguf_vlm": any(Path("~/.shuttersift/models").glob("*.gguf")),
    "api_vlm":  bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    "rawpy":    try_import("rawpy"),
}
```

---

## 4. CLI Interface

### 主命令

```bash
shuttersift <input_dir> [OPTIONS]

Arguments:
  input_dir    照片目录（递归扫描）

Options:
  --output     输出根目录（默认 <input_dir>/../shuttersift_output）
  --explain    对 Review 档启用 VLM 文字解释
  --dry-run    只分析不移动文件，输出报告
  --config     自定义 config.yaml 路径
  --keep-threshold    INT  覆盖 Keep 阈值（默认 70）
  --reject-threshold  INT  覆盖 Reject 阈值（默认 40）
  --workers    INT  并行 worker 数（默认 CPU 核数）
```

### 辅助命令

```bash
shuttersift download-models          # 下载 MediaPipe task 文件
shuttersift download-models --vlm    # 额外下载 moondream2 GGUF (~2GB)
shuttersift info                     # 显示已探测到的能力（GPU/VLM/RAW）
```

### 终端输出

```
ShutterSift v0.1.0

Scanning ./photos...  247 photos  (CR2: 198 · JPEG: 49)
Detected: GPU ✓  GGUF VLM ✓  RAW ✓

Analyzing  ████████████████████  247/247  [02:14]
VLM explaining Review photos...  ██████████  71/71  [01:03]

─────────────────────────────────────
  ✓  Keep      89  (36%)
  ◎  Review    71  (29%)
  ✗  Reject    87  (35%)
─────────────────────────────────────
Output  →  ./shuttersift_output/
Report  →  ./shuttersift_output/report.html
```

---

## 5. Output Structure

```
shuttersift_output/
├── keep/
│   ├── DSC_0042.CR2         (symlink to original)
│   └── DSC_0042.xmp         (Lightroom sidecar: Rating=5, Label=Green)
├── review/
│   └── ...
├── reject/
│   └── ...
├── report.html              # 可视化报告（缩略图+分数+理由）
└── results.json             # 机器可读结果
```

**XMP sidecar** 让 Lightroom 用户导入后直接按星级/颜色筛选，零额外操作。

---

## 6. Tech Stack

| 组件 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.11+ | — |
| CLI | typer + rich | 美观进度条，彩色输出 |
| 配置 | pydantic-settings + YAML | 默认值覆盖所有场景 |
| RAW 解码 | rawpy | CR2/NEF/ARW/DNG/RW2 全支持 |
| 图像处理 | opencv-python-headless + Pillow | headless，无 GUI 依赖 |
| 人脸/眼/笑容 | mediapipe (+ mediapipe-silicon on Apple Silicon) | 一个包，CPU，blendshapes |
| 审美评分 | pyiqa (MUSIQ) → BRISQUE fallback | 运行时自动选 |
| VLM 本地 | llama-cpp-python + GGUF | Metal/CUDA/CPU wheel 完整 |
| VLM 云端 | anthropic + openai | Review 档按需调用 |
| 去重 | imagehash | pHash，纯 CPU |
| 报告 | jinja2 | HTML 模板，离线可查 |
| 打包 | pyproject.toml (hatchling) | `pip install shuttersift` |

**注**：insightface 不纳入默认依赖（macOS M1/M2 编译问题）。MediaPipe 覆盖所有人脸分析需求。

---

## 7. Project Structure

```
ShutterSift/
├── src/shuttersift/
│   ├── __init__.py
│   ├── __main__.py          # typer CLI 入口
│   ├── config.py            # Pydantic 配置 + 默认阈值
│   ├── pipeline.py          # 主编排器（6 阶段）
│   ├── capabilities.py      # 运行时硬件/模型探测
│   ├── loader.py            # RAW/JPEG 统一加载
│   ├── scorer.py            # 加权合分 + 决策逻辑
│   ├── analyzers/
│   │   ├── sharpness.py
│   │   ├── exposure.py
│   │   ├── face.py          # MediaPipe 封装
│   │   ├── aesthetic.py     # MUSIQ / BRISQUE
│   │   ├── composition.py   # 规则引擎
│   │   └── duplicates.py   # pHash + burst 分组
│   ├── explainer.py         # GGUF / API VLM
│   ├── organizer.py         # 文件分类 + XMP sidecar
│   ├── reporter.py          # HTML + JSON 报告
│   └── downloader.py        # HF 模型下载
├── templates/
│   └── report.html.j2
├── docs/superpowers/specs/
│   └── 2026-04-02-shuttersift-design.md
├── tests/
├── pyproject.toml
├── config.yaml              # 默认配置示例
└── README.md
```

---

## 8. Key Design Decisions & Rationale

| 决策 | 理由 |
|------|------|
| MediaPipe 替代 insightface | macOS M1/M2 上 insightface 编译问题，MediaPipe wheel 完整且一包覆盖全部人脸需求 |
| 全量依赖，运行时降级 | 用户无需思考 extras，工具自适应环境 |
| GGUF via llama-cpp-python | 有 Metal/CUDA/CPU 预编译 wheel，无需 Docker，无需 PyTorch |
| VLM 仅处理 Review 档 | 控制成本和耗时，Keep/Reject 已有明确 CV 依据 |
| 早停机制 | Stage 2 硬拒绝后不进入后续阶段，提升吞吐 |
| symlink 而非 move | 非破坏性，原始文件安全 |
| XMP sidecar 输出 | Lightroom 零摩擦集成 |

---

## 9. Out of Scope (v0.1.0)

- GPU 批处理优化（v0.2）
- 个人风格 RAG / few-shot 学习（v0.3）
- Gradio Web UI（v0.3）
- 自然语言查询（v0.4）
- Lightroom SDK 直接写入 catalog（v0.4）
