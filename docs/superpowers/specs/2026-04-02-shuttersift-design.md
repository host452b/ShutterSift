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

### 7.1 分层架构原则

```
┌─────────────────────────────────────────────┐
│            Presentation Layer               │
│  CLI (typer)  │  Server (FastAPI, future)   │
└───────────────┼─────────────────────────────┘
                │  调用 Python API
┌───────────────▼─────────────────────────────┐
│              Engine Layer                   │
│  pipeline · analyzers · scorer · explainer  │
│  (纯逻辑，无 UI 依赖，可独立测试)              │
└───────────────┬─────────────────────────────┘
                │  IPC (localhost HTTP, future)
┌───────────────▼─────────────────────────────┐
│             GUI Clients (future)            │
│   desktop (Tauri/Electron)  │  web          │
└─────────────────────────────────────────────┘
```

Engine 层对所有客户端暴露统一的 Python API（`shuttersift.engine`），CLI 是第一个客户端，未来 GUI 通过本地 FastAPI server 调用同一 engine。

### 7.2 目录结构

```
ShutterSift/
├── src/shuttersift/
│   │
│   ├── engine/                  # ★ 核心引擎，无 UI 依赖
│   │   ├── __init__.py          #   暴露 Engine 公共 API
│   │   ├── pipeline.py          #   主编排器（6 阶段）
│   │   ├── capabilities.py      #   运行时硬件/模型探测
│   │   ├── loader.py            #   RAW/JPEG 统一加载
│   │   ├── scorer.py            #   加权合分 + 决策逻辑
│   │   ├── organizer.py         #   文件分类 + XMP sidecar
│   │   ├── reporter.py          #   HTML + JSON 报告
│   │   ├── downloader.py        #   HF 模型下载 + SHA256 校验
│   │   ├── explainer.py         #   GGUF / API VLM
│   │   ├── state.py             #   断点续传 .state.json 管理
│   │   └── analyzers/
│   │       ├── sharpness.py
│   │       ├── exposure.py
│   │       ├── face.py
│   │       ├── aesthetic.py
│   │       ├── composition.py
│   │       └── duplicates.py
│   │
│   ├── cli/                     # CLI 表现层
│   │   ├── __init__.py
│   │   └── main.py              #   typer app，调用 engine API
│   │
│   ├── server/                  # ★ 预留：本地 HTTP server（GUI 桥接层）
│   │   ├── __init__.py          #   空，占位
│   │   ├── app.py               #   FastAPI app 骨架（注释状态）
│   │   └── README.md            #   说明：GUI 客户端通过此 server 与 engine 通信
│   │
│   ├── config.py                # Pydantic 配置模型 + 默认阈值
│   └── __init__.py
│
├── clients/                     # ★ 预留：GUI 客户端
│   ├── desktop/                 #   未来 Tauri / Electron 桌面应用
│   │   └── README.md            #   说明：调用 shuttersift server 的桌面客户端
│   └── web/                     #   未来 Web UI（Gradio / 自定义）
│       └── README.md
│
├── templates/
│   └── report.html.j2
├── docs/superpowers/specs/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                # 测试用小图（含各类问题照片样本）
├── pyproject.toml
├── config.yaml
└── README.md
```

### 7.3 engine 公共 API 契约（CLI 和未来 GUI 共用）

```python
# src/shuttersift/engine/__init__.py
from .pipeline import Engine, AnalysisResult, PhotoResult

class Engine:
    def __init__(self, config: Config): ...
    
    def analyze(
        self,
        input_dir: Path,
        output_dir: Path,
        on_progress: Callable[[int, int, PhotoResult], None] | None = None,
        resume: bool = True,
    ) -> AnalysisResult: ...
    
    def capabilities(self) -> dict: ...   # GPU/VLM/RAW 探测结果
```

`on_progress` 回调使 CLI 能更新进度条、未来 GUI 能推送 WebSocket 事件——**接口不变，展示层自由替换**。

---

## 8. Commercial-Grade Requirements

### 8.1 断点续传

每张照片完成分析后立即写入 `output/.state.json`。下次运行检测到 state 文件时默认 resume，用 `--fresh` 强制重跑。state 文件包含：文件路径、mtime、综合分、决策、各子分。

### 8.2 重复运行策略

- 默认：检测到 `output/.state.json` → 跳过已处理文件，只处理新增/修改文件
- `--fresh`：忽略 state，全量重跑，覆盖输出
- 输出目录已存在但无 state 文件：警告用户，不自动覆盖

### 8.3 清晰度阈值自动校准

固定阈值 30 在不同相机/镜头上误差大。解决方案：

- 首次运行时采样输入目录中最清晰的 5% 照片，计算 Laplacian 分布
- 将 `hard_reject` 阈值设为 `p5`（第 5 百分位），`soft_warn` 设为 `p20`
- 校准结果缓存到 `~/.shuttersift/calibration/<camera_model>.json`
- 用户可运行 `shuttersift calibrate ./photos` 手动触发

### 8.4 Apple Silicon 安装

`pyproject.toml` 使用 platform marker 自动选包：

```toml
dependencies = [
  "mediapipe; platform_machine != 'arm64'",
  "mediapipe-silicon; platform_machine == 'arm64'",
  ...
]
```

### 8.5 结构化日志

每次运行写入 `~/.shuttersift/logs/YYYY-MM-DDTHH-MM-SS.log`，内容包含：
- 每张照片各子分原始值
- 使用的降级路径（MUSIQ/BRISQUE、GGUF/API/无）
- 异常堆栈（文件损坏、模型加载失败等）
- 各阶段耗时（便于性能排查）

日志保留最近 30 次，自动清理。

### 8.6 模型完整性校验

`downloader.py` 下载模型后验证 SHA256，失败则自动重试（最多 3 次）后报错。校验摘要硬编码在 `downloader.py` 中，随版本更新。

### 8.7 results.json 版本化

```json
{
  "version": "1",
  "shuttersift_version": "0.1.0",
  "run_at": "2026-04-02T14:30:00Z",
  "photos": [...]
}
```

未来版本新增字段时 `version` 递增，读取器向后兼容。

---

## 9. Key Design Decisions & Rationale

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
