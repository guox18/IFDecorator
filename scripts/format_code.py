#!/usr/bin/env python3
"""
代码格式化脚本
自动使用 black、isort 和 flake8 来美化代码风格
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并显示结果"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} 完成")
        if result.stdout.strip():
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        if e.stdout.strip():
            print(f"   输出: {e.stdout.strip()}")
        if e.stderr.strip():
            print(f"   错误: {e.stderr.strip()}")
        return False


def main():
    """主函数"""
    print("🎨 开始格式化代码...")
    
    # 确保在项目根目录
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 定义要格式化的目录
    directories = ["modules", "infer", "recipe", "tripwires", "*.py"]
    
    success_count = 0
    total_steps = 3
    
    # 1. 使用 isort 排序 imports
    for directory in directories:
        if run_command(
            ["python", "-m", "isort", directory],
            f"排序 {directory} 的 imports"
        ):
            success_count += 1
    
    # 2. 使用 black 格式化代码
    for directory in directories:
        if run_command(
            ["python", "-m", "black", directory],
            f"格式化 {directory} 的代码"
        ):
            success_count += 1
    
    # 3. 使用 flake8 检查代码质量
    print("🔍 检查代码质量...")
    try:
        result = subprocess.run(
            ["python", "-m", "flake8", "--statistics"] + directories,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("✅ 代码质量检查通过！")
        else:
            print("⚠️  发现一些代码质量问题:")
            print(result.stdout)
    except Exception as e:
        print(f"❌ 代码质量检查失败: {e}")
    
    print(f"\n🎉 代码格式化完成！")
    print("💡 建议:")
    print("   - 使用 'python scripts/format_code.py' 定期格式化代码")
    print("   - 在提交前运行此脚本确保代码风格一致")
    print("   - 考虑配置 pre-commit hooks 自动化这个过程")


if __name__ == "__main__":
    main() 