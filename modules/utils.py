"""
Utility functions for the IFDecorator project.
This module provides a collection of helper functions for file operations,
text processing, data manipulation, and progress tracking.
"""

import copy
import json
import os
import pickle
import random
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from langdetect import detect
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from global_config import PROJECT_ROOT

sys.path.append(PROJECT_ROOT)

from logs.logger import logger


def unified_judge_parse(response: str, flag_strict: bool = False) -> bool:
    """
    Parse and judge the verification response from the model.

    Args:
        response (str): The model's response text
        flag_strict (bool): Whether to use strict matching rules

    Returns:
        bool: True if verification passes, False otherwise
    """
    if response is None:
        return False

    if flag_strict:
        if (
            ("**Final Verification:** YES" in response)
            or ("Final Verification: YES" in response)
            or ("Verification: YES" in response)
        ):
            return True
        else:
            return False


def chance(probability):
    """
    Return True with the given probability.

    Args:
        probability (float): Probability between 0 and 1

    Returns:
        bool: True with probability 'probability', False otherwise
    """
    return random.random() < probability


def vis_print(seq):
    """
    Print a sequence with decorative separators for better visibility.

    Args:
        seq: The sequence to print
    """
    print("." * 20)
    print(seq)
    print("`" * 20)


def seed_everything(seed):
    """
    Set random seeds for reproducibility across all random number generators.

    Args:
        seed (int): The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def split_list(lst, batch_parallel):
    """
    Split a list into approximately equal-sized batches.

    Args:
        lst (list): The list to split
        batch_parallel (int): Number of batches to create

    Returns:
        list: List of sub-lists
    """
    # 计算每个子列表的大小
    avg_size = len(lst) // batch_parallel
    remainder = len(lst) % batch_parallel

    batches = []
    start = 0

    for i in range(batch_parallel):
        # 如果还有余数，将多余的元素分配到前面的子列表
        end = start + avg_size + (1 if i < remainder else 0)
        batches.append(lst[start:end])
        start = end

    return batches


def get_current_time_as_string():
    """
    Get current timestamp as a formatted string.

    Returns:
        str: Current time in format 'YYYYMMDDHHMMSSffffff'
    """
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def writejsonl(data, datapath):
    """
    Write data to a JSONL file.

    Args:
        data (list): List of JSON-serializable objects
        datapath (str): Path to the output file
    """
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    logger.info(f"saving file at {datapath}")
    with open(datapath, "w", encoding="utf-8") as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")


def writejson(data, json_path):
    """
    Write data to a JSON file with pretty printing.

    Args:
        data: JSON-serializable object
        json_path (str): Path to the output file
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    logger.info(f"saving file at {json_path}")
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)


def readjsonl(datapath):
    """
    Read data from a JSONL file.

    Args:
        datapath (str): Path to the JSONL file

    Returns:
        list: List of objects from the JSONL file
    """
    res = []
    logger.info(f"reading file at {datapath}")
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def readjson(datapath):
    """
    Read data from a JSON file.

    Args:
        datapath (str): Path to the JSON file

    Returns:
        The parsed JSON data
    """
    logger.info(f"reading file at {datapath}")
    with open(datapath, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


def writepickle(data, datapath):
    """
    Write data to a pickle file.

    Args:
        data: Object to pickle
        datapath (str): Path to the output file
    """
    with open(datapath, "wb") as f:
        pickle.dump(data, f)


def readpickle(datapath):
    """
    Read data from a pickle file.

    Args:
        datapath (str): Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(datapath, "rb") as f:
        res = pickle.load(f)
    return res


def readlargepickle(load_file):
    """
    Generator to read large pickle files incrementally.

    Args:
        load_file (str): Path to the pickle file

    Yields:
        Objects from the pickle file one at a time
    """
    with open(load_file, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def check_folder(path):
    """
    Create a folder if it doesn't exist.

    Args:
        path (str): Path to the folder
    """
    if not os.path.exists(path):
        logger.info(f"{path} not exists, create it")
        os.makedirs(path)


def get_name(name, pattern, mode=0):
    """
    Extract content from a string using regex pattern.

    Args:
        name (str): String to search in
        pattern (str): Regex pattern to match
        mode (int): Group number to extract from match

    Returns:
        str: Extracted content or None if no match
    """
    match = re.search(pattern, name)
    # 提取结果
    if match:
        extracted_content = match.group(mode)
        return extracted_content
    else:
        logger.info("Pattern not found")
    x = json.JSONEncode()


def is_en_lang(text):
    """
    Check if text is in English language.

    Args:
        text (str): Text to check

    Returns:
        bool: True if text is in English, False otherwise
    """
    try:
        return detect(text) == "en"
    except:
        # 异常情况（如极短文本或乱码）默认返回 "Yes"
        logger.warning(f"异常情况（如极短文本或乱码）默认返回 False")
        return False


def read_and_print_json(datapath, num1=0, num2=5):
    """
    Read and print a range of items from a JSON/JSONL file.

    Args:
        datapath (str): Path to the JSON/JSONL file
        num1 (int): Start index
        num2 (int): End index

    Returns:
        The loaded data
    """
    ext = os.path.splitext(datapath)[1]
    data = readjsonl(datapath) if ext == ".jsonl" else readjson(datapath)
    for i in range(num1, min(num2, len(data))):
        logger.info(f"Item {i}:", data[i])
    return data


if __name__ == "__main__":
    total_items = 100

    # 自定义进度条样式
    with Progress(
        SpinnerColumn(),  # 显示一个动态的加载动画
        TextColumn("[bold blue]{task.description}"),  # 任务描述，加粗蓝色
        BarColumn(
            bar_width=40,
            complete_style="green",
            finished_style="green",
            pulse_style="yellow",
        ),  # 进度条
        TaskProgressColumn(),  # 显示百分比进度
        TextColumn("({task.completed}/{task.total})", justify="right"),  # 当前项/总项数
        TimeRemainingColumn(),  # 显示剩余时间 (ETA)
        transient=True,  # 完成后自动清除
    ) as progress:
        # 添加一个任务，指定总项数
        task = progress.add_task("[cyan]Processing...", total=total_items)

        # 模拟处理每一项
        for i in range(total_items):
            # 模拟一些处理逻辑
            progress.update(task, advance=1)  # 更新进度条
