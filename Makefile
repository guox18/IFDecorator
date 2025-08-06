.PHONY: help format lint test install-dev setup-hooks clean

# 默认目标
help:
	@echo "🎨 代码美化工具使用指南"
	@echo ""
	@echo "可用命令:"
	@echo "  make install-dev    - 安装开发依赖"
	@echo "  make setup-hooks    - 设置git pre-commit hooks"
	@echo "  make format         - 格式化所有代码"
	@echo "  make lint           - 检查代码质量"
	@echo "  make test           - 运行测试"
	@echo "  make clean          - 清理临时文件"

# 安装开发依赖
install-dev:
	@echo "📦 安装开发依赖..."
	pip install -e ".[dev]"
	pip install pre-commit

# 设置git hooks
setup-hooks:
	@echo "🔧 设置pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks已安装"

# 格式化代码
format:
	@echo "🎨 格式化代码..."
	python -m isort modules/ infer/ recipe/ tripwires/ *.py
	python -m black modules/ infer/ recipe/ tripwires/ *.py
	@echo "✅ 代码格式化完成"

# 代码质量检查
lint:
	@echo "🔍 检查代码质量..."
	python -m flake8 modules/ infer/ recipe/ tripwires/ *.py --statistics
	@echo "✅ 代码质量检查完成"

# 运行格式化脚本
format-script:
	@echo "🚀 运行格式化脚本..."
	python scripts/format_code.py

# 快速格式化（仅当前修改的文件）
format-staged:
	@echo "⚡ 格式化已暂存的文件..."
	pre-commit run --files $$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$$' || echo "")

# 运行测试
test:
	@echo "🧪 运行测试..."
	python -m pytest tests/ -v

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	@echo "✅ 清理完成"

# 完整的代码质量检查
check-all: format lint
	@echo "🎉 所有代码质量检查完成！" 