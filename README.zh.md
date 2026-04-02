# ShutterSift

AI 驱动的照片筛选工具。丢一个文件夹进去，自动分出「保留 / 待定 / 淘汰」。

```bash
ss ./photos
```

每张照片都会获得一个 0–100 的综合评分，覆盖五个质量维度。结果自动整理进
子目录，并生成可交互的 HTML 报告。

---

## 安装

**从 PyPI 安装**（发布后可用）：
```bash
pip install shuttersift
```

**从 GitHub Release 安装**（现在可用）：
```bash
pip install https://github.com/host452b/ShutterSift/releases/download/v0.1.0/shuttersift-0.1.0-py3-none-any.whl
```

**从源码安装**：
```bash
pip install git+https://github.com/host452b/ShutterSift.git
```

> **Apple Silicon（M1/M2/M3）**：如果 `mediapipe` 安装失败，请改用：
> ```bash
> pip install mediapipe-silicon
> pip install shuttersift --no-deps
> ```

> **Mac GPU 加速（Metal）**：
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
> ```

---

## 快速开始

```bash
# 第一步 — 下载所需模型文件（只需一次）
ss setup

# 第二步 — 扫描你的照片
ss ./photos

# 第三步 — 查看报告
open shuttersift_output/report.html      # macOS
xdg-open shuttersift_output/report.html  # Linux
```

首次运行时会自动根据你的照片库校准锐度阈值，**无需任何手动配置**。

`ss` 和 `shuttersift` 都是合法的命令名，两者等价，用哪个都行。

---

## 输出结构

```
shuttersift_output/
├── keep/         ← 高分照片（符号链接 + XMP 附属文件）
├── review/       ← 边界照片，值得二次确认
├── reject/       ← 模糊、闭眼、重复照片
├── report.html   ← 带评分和缩略图的可交互报告
└── results.json  ← 机器可读，带版本号
```

**Lightroom 用户**：直接导入 `keep/` 文件夹，XMP 附属文件会自动设置星级和色标。

---

## 常用参数

```bash
ss ./photos -e               # 对「待定」照片启用 VLM 解释
ss ./photos -n               # 演习模式——只分析，不写入文件
ss ./photos -f               # 强制重新分析（忽略缓存）
ss ./photos --recalibrate    # 对当前图库重新校准锐度
ss ./photos -o ./sorted      # 指定输出目录
ss ./photos --keep 75 --reject 35   # 自定义评分阈值
ss ./photos -j 8             # 使用 8 个并行 worker
ss ./photos -v               # 详细日志输出
```

| 短参数 | 长参数 | 默认值 | 说明 |
|--------|--------|--------|------|
| `-e` | `--explain` | 关 | 对边界照片启用 VLM 文字解释 |
| `-n` | `--dry-run` | 关 | 只分析，不移动或链接文件 |
| `-f` | `--force` | 关 | 忽略缓存，重新分析所有照片 |
| `-o` | `--output <目录>` | `../shuttersift_output` | 输出目录 |
| `-j` | `--jobs <n>` | 4 | 并行 worker 数量 |
| `-v` | `--verbose` | 关 | 向 stderr 打印调试日志 |
| `-c` | `--config <文件>` | 自动 | 指定配置文件路径 |
| | `--keep <n>` | 70 | 保留所需最低分数 |
| | `--reject <n>` | 40 | 低于此分数直接淘汰 |
| | `--recalibrate` | 关 | 强制重新运行锐度校准 |

---

## VLM 解释功能

加上 `-e` / `--explain` 参数后，落在「待定」区间的边界照片会获得 AI 生成的
文字描述，解释问题所在或值得关注的点。通常有 20–30% 的照片落在待定区间。

**云端 API（Anthropic 或 OpenAI）**：
```bash
export ANTHROPIC_API_KEY=sk-ant-...
ss ./photos -e
```
```bash
export OPENAI_API_KEY=sk-...
ss ./photos -e
```

**完全本地运行（不联网）**：
```bash
ss setup --vlm        # 下载 moondream2 GGUF 模型，约 1.7 GB，仅需一次
ss ./photos -e        # 使用本地模型，无需 API Key
```

---

## 评分体系

每张照片获得一个 0–100 的综合评分：

| 维度 | 权重 | 方法 |
|------|------|------|
| 锐度 | 30% | Laplacian 方差 |
| 曝光 | 15% | 直方图分析 |
| 美学 | 25% | MUSIQ（GPU）/ BRISQUE（CPU）|
| 人脸质量 | 20% | MediaPipe 睁眼度 + 笑容检测 |
| 构图 | 10% | 三分法引擎 |

**默认阈值**：≥ 70 → 保留 · 40–69 → 待定 · < 40 → 淘汰

通过 `--keep` 和 `--reject` 自定义阈值。

---

## 高级用法

### 配置文件

ShutterSift 按以下顺序查找配置文件：
1. `--config <路径>`（显式指定）
2. `./shuttersift.yaml`（当前目录）
3. `./config.yaml`（当前目录）
4. `~/.shuttersift/config.yaml`（用户全局）

完整配置参考（`~/.shuttersift/config.yaml`）：

```yaml
calibrated: true               # 首次运行后自动设置

scoring:
  thresholds:
    keep: 70                   # 保留所需最低分数
    reject: 40                 # 低于此分数直接淘汰
    hard_reject_sharpness: 42.3  # 由自动校准写入
    eye_open_min: 0.25         # 最低睁眼比例（0–1）
    burst_gap_seconds: 2.0     # 此时间窗口内的连拍视为同一组

workers: 4                     # 并行 worker 数量
log_retention_runs: 30         # 保留的日志文件数量
```

### 手动校准

```bash
ss calibrate ./photos
```

采样最多 300 张照片，打印完整的 Laplacian 方差分布，
并将推荐的 `hard_reject_sharpness` 值保存到 `~/.shuttersift/config.yaml`。
切换相机或拍摄风格时使用。

### 其他命令

```bash
ss info          # 查看 GPU、VLM 和 RAW 支持状态
ss setup --vlm   # 下载本地 moondream2 VLM 模型
```

### GUI 客户端（规划中）

桌面端和 Web 端 GUI 客户端正在规划中。
参见 `clients/desktop/` 和 `clients/web/` 目录中的早期骨架。

---

## English README

[English Documentation →](README.md)
