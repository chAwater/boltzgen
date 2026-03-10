# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

BoltzGen 是一个蛋白质设计工具，基于扩散模型生成蛋白质结构，集成反向折叠、共折叠验证和亲和力预测。核心模型基于 PairFormer + 三角形注意力架构（类 AlphaFold），使用 PyTorch Lightning 训练。

## 常用命令

```bash
# 安装
pip install -e .           # 开发模式
pip install -e ".[dev]"    # 包含 lint/test/wandb 等

# 测试
pytest tests/
pytest --mock-heavy-deps tests/  # Mock GPU/科学计算依赖，仅测试解析逻辑

# Lint
ruff check src/
ruff format src/

# 运行管道
boltzgen run design.yaml --out output/
boltzgen check design.yaml        # 仅验证设计规范
boltzgen configure design.yaml    # 生成配置但不执行
```

## 代码架构

### 管道流程（顺序执行）

design(GPU) → inverse_folding(GPU) → design_folding(GPU,可选) → folding(GPU) → affinity(GPU,可选) → analysis(CPU) → filtering(CPU)

### 核心模块

- **`cli/boltzgen.py`** — CLI 入口（~77KB），定义子命令 `run/check/configure/execute/download/merge`，管理管道步骤编排和配置组装
- **`task/`** — 管道步骤实现，继承 `Task` 抽象基类（`task.py`），每个步骤通过 `run(config)` 执行
  - `predict/` — GPU 步骤：扩散生成、反向折叠、折叠、亲和力
  - `analyze/` — CPU：设计质量指标计算
  - `filter/` — CPU：排序和多样性选择
- **`data/parse/schema.py`** — YAML 设计规范解析器（`YamlDesignParser`），处理残基约束、二级结构、结合位点等规范
- **`model/`** — 深度学习模型
  - `models/boltz.py` — 主模型（LightningModule）
  - `layers/` — PairFormer、三角形注意力、MiniformerModule 等
  - `loss/` — 扩散损失、置信度损失
- **`data/`** — 数据处理：核心数据类在 `data.py`（Atom, Residue, Chain, Structure），常数在 `const.py`
- **`resources/config/`** — Hydra YAML 配置文件，通过 `resources/main.py` 实例化 Task

### 配置管理

使用 Hydra 框架管理配置。每个管道步骤对应 `resources/config/` 下的 YAML 文件。CLI 通过 `--config key=value` 覆盖参数。

### 设计规范（YAML）

用户通过 YAML 文件定义设计目标，包含 `entities`（蛋白质/配体/文件）和 `constraints`（化学键/长度约束）。支持序列长度范围、二级结构约束、结合位点规范等。示例见 `example/` 目录。

### 内置协议

`protein-anything`, `peptide-anything`, `protein-small_molecule`, `nanobody-anything`, `antibody-anything`, `protein-redesign` — 每个协议预设不同的管道参数和指标。

### 数据序列化

- NPZ — 数值数据（`NumpySerializable`）
- JSON — 元数据（`JSONSerializable`）
- mmCIF — 结构输出（`to_mmcif`）

## 代码规范

- Ruff 严格模式（`select = ["ALL"]`），NumPy docstring 风格
- MyPy 要求函数类型注解（`disallow_untyped_defs = true`）
- Python >= 3.11
