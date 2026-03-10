# AGENTS.md

本文件为在 boltzgen 代码库中工作的 AI Agent 提供开发规范指导。

## 项目概述

BoltzGen 是一个基于扩散模型和 Boltz-2 折叠的蛋白质binder设计系统。生成流程：design → inverse folding → folding → analysis → filtering。

## 学习文档

项目包含完整的学习文档系统（位于 `docs/`），采用原子化笔记 + MOC 索引结构：
- **35 个原子化笔记**：每个聚焦单一主题，50-150 行
- **4 个 MOC 索引**：数据层、模型层、训练系统、推理系统
- **完整双向链接网络**：Obsidian 兼容
- **学习顺序编号**：01、02、03... 清晰指引

详见：
- `docs/README.md` - 主导航
- `docs/DIALOGUE_LOG.md` - 文档创建方法论

## 构建与测试命令

```bash
# 安装 (开发模式)
pip install -e .[dev]

# Lint 检查
ruff check src/
ruff check src/boltzgen/data/data.py    # 检查单个文件
ruff check --fix src/                   # 自动修复

# 代码格式化
ruff format src/

# 类型检查
mypy src/
mypy src/boltzgen/data/data.py          # 检查单个文件

# 运行测试
pytest                                  # 运行所有测试
pytest tests/test_specific.py           # 运行单个测试文件
pytest tests/test_specific.py::test_func # 运行单个测试函数
pytest -k "test_name"                   # 按名称过滤测试

# CLI 命令
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 --budget 2

# 验证设计规范 (不运行)
boltzgen check example/vanilla_protein/1g13prot.yaml
```

## 代码风格指南

### 导入规范

```python
# 标准库 → 第三方库 → 本地模块
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit.Chem import Mol

from boltzgen.data import const
from boltzgen.data.data import Structure
```

### 格式化规则

- **行长度**: 无硬性限制，但保持合理
- **缩进**: 4 空格
- **引号**: 双引号优先 (`"string"`)
- **尾随逗号**: 保持
- **空行**: 两个空行分隔顶级定义 (类/函数)

### 类型注解

- **必须启用**: `disallow_untyped_defs = true` (mypy)
- **使用泛型**: `List[str]`, `Dict[str, int]`, `Optional[str]`
- **复杂类型**: 使用 `Union` 而非 `|`，保持兼容性
- **Future annotations**: 项目不强制使用 `from __future__ import annotations`

```python
def process_structure(
    structure: Structure,
    config: Dict[str, int],
) -> Optional[Structure]:
    ...
```

### 命名约定

| 类型 | 规则 | 示例 |
|------|------|------|
| 模块 | 小写下划线 | `data_parser.py` |
| 类 | 大驼峰 | `class Structure:` |
| 函数/变量 | 小写下划线 | `def get_positions():` |
| 常数 | 全大写下划线 | `MAX_RESIDUES = 10000` |
| 私有成员 | 单下划线前缀 | `_private_method()` |

### Docstring 风格

使用 **NumPy 风格** docstring：

```python
class Structure:
    """A protein structure representation.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms in the structure.
    chains : List[Chain]
        List of chains in the structure.

    Attributes
    ----------
    num_residues : int
        Number of residues in the structure.

    """

    def get_atom(self, chain_id: str, res_id: int) -> Optional[Atom]:
        """Get an atom by chain and residue ID.

        Parameters
        ----------
        chain_id : str
            The chain identifier.
        res_id : int
            The residue identifier (1-based).

        Returns
        -------
        Optional[Atom]
            The atom if found, None otherwise.

        """
        ...
```

### 错误处理

- **异常消息**: 使用通用描述，避免在异常中暴露敏感信息
- **异常类型**: 使用标准异常或项目自定义异常
- **捕获**: 避免空的 except 块，指定具体异常类型

```python
try:
    result = load_structure(path)
except FileNotFoundError:
    logger.warning(f"Structure file not found: {path}")
    return None
```

### Ruff 规则忽略

以下规则被显式忽略：

| 规则 | 原因 |
|------|------|
| COM812 | 与 formatter 冲突 |
| ANN101 | 不强制 self 类型注解 |
| S101 | 允许 assert (测试用) |
| D100, D104 | 允许无 docstring 的模块/包 |
| PT001, PT004, PT005, PT023 | pytest 风格规则 |
| FBT001, FBT002 | 允许布尔类型参数 |
| PLR0913 | 允许超过 5 个参数 |
| TRY003, EM101 | 允许异常消息 |
| FA102 | 允许 `from __future__ import annotations` |
| T201 | 允许 print 语句 |

### 特定文件规则

- `tests/`: 允许 assert、S103、允许无 docstring
- `__init__.py`: 允许未使用的导入、通配符导入
- `docs/`、`scripts/`: 允许无 `__init__.py`

### 关键约定

- **残基索引**: 1-based，使用 `label_asym_id` (非 `auth_asym_id`)
- **YAML 路径**: 相对于 YAML 文件所在目录
- **模型缓存**: 下载到 `~/.cache`，可通过 `--cache` 或 `$HF_HOME` 配置

### 代码组织

```
src/boltzgen/
├── cli/              # CLI 入口
├── data/             # 数据层 (解析、特征、写入)
│   ├── parse/        # 结构文件解析
│   ├── tokenize/     # token 化
│   └── write/        # 输出格式
├── model/            # 模型层 (PyTorch Lightning)
│   ├── models/       # 主模型
│   ├── modules/      # 核心模块
│   ├── layers/       # 基础层
│   └── loss/         # 损失函数
├── task/             # 任务系统 (Predict/Analyze/Filter/Train)
└── utils/            # 工具函数
```

### 开发注意事项

- 所有代码必须通过 `ruff check` 和 `ruff format`
- 类型注解必须完整 (mypy 强制)
- 提交前运行完整 lint 和类型检查