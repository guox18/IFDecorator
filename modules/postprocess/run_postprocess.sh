#!/bin/bash

# CIF后处理流程脚本
# 用法: ./run_postprocess.sh [pipeline_num] [input_file]
#
# 参数说明:
# pipeline_num: 管道编号，用于指定数据管道路径 (可选，默认等待用户传入)
# input_file: 输入的JSON文件路径 (可选，如果不提供将使用gather.py收集的数据)
#
# 流程步骤:
# 1. gather.py - 收集所有hard_pool中的数据，按reasonable/unreasonable分类
# 2. post_filter.py - 对数据进行分类过滤(Math/Code/Reasoning/Other)
# 3. trans2verl_format.py - 转换为VERL格式并保存为jsonl和parquet格式

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 颜色输出函数
print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 参数解析
PIPELINE_NUM=$1
INPUT_FILE=$2

# 检查参数
if [ -z "$PIPELINE_NUM" ]; then
    print_warn "未提供pipeline_num参数，请设置环境变量或修改代码中的pipeline_num配置"
    print_info "示例用法: ./run_postprocess.sh 1 [input_file]"
fi

if [ -z "$INPUT_FILE" ]; then
    print_info "未提供input_file参数，将使用gather.py收集数据"
    USE_GATHER=true
else
    print_info "使用指定的输入文件: $INPUT_FILE"
    USE_GATHER=false
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        print_error "输入文件不存在: $INPUT_FILE"
        exit 1
    fi
fi

print_info "开始CIF后处理流程..."
print_info "项目根目录: $PROJECT_ROOT"
print_info "脚本目录: $SCRIPT_DIR"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 步骤1: 数据收集 (如果需要)
if [ "$USE_GATHER" = true ]; then
    print_info "步骤1/3: 执行数据收集 (gather.py)"
    print_info "收集所有管道产出的数据，按reasonable/unreasonable分类"
    
    if [ -n "$PIPELINE_NUM" ]; then
        export PIPELINE_NUM="$PIPELINE_NUM"
    fi
    
    python gather.py
    
    if [ $? -eq 0 ]; then
        print_info "数据收集完成 ✓"
        # 使用收集的reasonable数据作为下一步的输入
        COLLECTED_DATA="$PROJECT_ROOT/paper/experiments/datasets/${PIPELINE_NUM}/cif/reasonable_dataset.jsonl"
        INPUT_FILE="$COLLECTED_DATA"
    else
        print_error "数据收集失败 ✗"
        exit 1
    fi
else
    print_info "跳过数据收集步骤 (使用指定的输入文件)"
fi

# 步骤2: 数据过滤
print_info "步骤2/3: 执行数据过滤 (post_filter.py)"
print_info "对数据进行分类: Math Problem, Code Task, Reasoning Task, Other"

python post_filter.py --input_file "$INPUT_FILE"

if [ $? -eq 0 ]; then
    print_info "数据过滤完成 ✓"
    # 获取过滤后的clean数据路径
    FILTER_DIR="$(dirname "$INPUT_FILE")/filtered"
    CLEAN_DATA="${FILTER_DIR}/$(basename "$INPUT_FILE" .jsonl)/clean.jsonl"
else
    print_error "数据过滤失败 ✗"
    exit 1
fi

# 步骤3: 格式转换
print_info "步骤3/3: 执行格式转换 (trans2verl_format.py)"
print_info "转换为VERL训练格式，生成jsonl和parquet文件"

if [ -n "$PIPELINE_NUM" ]; then
    export PIPELINE_NUM="$PIPELINE_NUM"
fi

python trans2verl_format.py

if [ $? -eq 0 ]; then
    print_info "格式转换完成 ✓"
else
    print_error "格式转换失败 ✗"
    exit 1
fi

# 完成提示
print_info "🎉 CIF后处理流程完成!"
print_info ""
print_info "生成的文件:"
if [ "$USE_GATHER" = true ]; then
    print_info "  - 收集的数据: $PROJECT_ROOT/paper/experiments/datasets/${PIPELINE_NUM}/cif/reasonable_dataset.jsonl"
    print_info "  - 收集的数据: $PROJECT_ROOT/paper/experiments/datasets/${PIPELINE_NUM}/cif/unreasonable_dataset.jsonl"
fi
print_info "  - 过滤结果: $FILTER_DIR/"
print_info "    ├── math_filtered.jsonl    (数学问题)"
print_info "    ├── code_filtered.jsonl    (编程任务)"
print_info "    ├── logic_filtered.jsonl   (推理任务)"
print_info "    └── clean.jsonl           (其他清洁数据)"
print_info "  - VERL格式: $PROJECT_ROOT/paper/experiments/datasets/${PIPELINE_NUM}/cif/verl_format/"
print_info "    ├── train_verl_format_dataset.jsonl"
print_info "    ├── test_verl_format_dataset.jsonl"
print_info "    ├── train_verl_format_dataset.parquet"
print_info "    └── test_verl_format_dataset.parquet"
print_info ""
print_info "可以使用以下命令查看结果:"
print_info "  head -n 3 $CLEAN_DATA"