# 代码风格指南

本项目采用统一的代码风格来确保代码的可读性和一致性。

## 🛠️ 工具链

### 核心工具
- **Black**: Python代码格式化器
- **isort**: Import语句排序工具  
- **flake8**: 代码质量检查工具
- **pre-commit**: Git钩子管理工具

### 安装和设置

1. **安装开发依赖**:
   ```bash
   make install-dev
   ```

2. **设置Git钩子**:
   ```bash
   make setup-hooks
   ```

## 🎨 使用方法

### 一键格式化
```bash
# 格式化所有代码
make format

# 或使用我们的自定义脚本
python scripts/format_code.py
```

### 代码检查
```bash
# 检查代码质量
make lint

# 完整的格式化+检查
make check-all
```

### 自动化工作流
- **提交前自动格式化**: 设置好pre-commit hooks后，每次`git commit`时会自动格式化代码
- **仅格式化修改的文件**: `make format-staged`

## 📋 代码风格规范

### 基本规则
- **行长度**: 最大88字符
- **缩进**: 4个空格
- **引号**: 优先使用双引号
- **Import顺序**: 标准库 → 第三方库 → 本地模块

### Import顺序示例
```python
# 标准库
import os
import sys
from pathlib import Path

# 第三方库
import numpy as np
import torch
from transformers import AutoModel

# 本地模块
from modules.utils import helper_function
from global_config import PROJECT_ROOT
```

### 函数定义规范
```python
def function_name(param1: str, param2: int = 0) -> bool:
    """
    函数功能简述
    
    Args:
        param1 (str): 参数说明
        param2 (int, optional): 可选参数说明. Defaults to 0.
    
    Returns:
        bool: 返回值说明
    """
    return True
```

## 💡 最佳实践

1. **开发前**: 运行`make install-dev`和`make setup-hooks`
2. **编码时**: 使用IDE的格式化插件
3. **提交前**: 运行`make check-all`确保代码质量
4. **定期维护**: 使用`make clean`清理临时文件 