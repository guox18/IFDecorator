"""
Multiprocessing wrapper module for parallel task execution.

This module provides classes and functions for managing parallel processing tasks
with checkpoint support and progress tracking. It includes:
- CheckpointManager: For managing task progress and recovery
- MultiProcessHandler: For parallel task execution
- Helper functions for data preprocessing and parallel execution
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from tqdm import tqdm
import concurrent.futures

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

console = Console()

import json
import os
import time
from datetime import datetime

from colorama import Fore, Style

from logs.logger import logger


def writejson(data, datapath):
    """
    Write data to a JSON file with pretty printing.

    Args:
        data: Data to serialize to JSON
        datapath: Path to the output file
    """
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    logger.info(f"Saving file at {datapath}")
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(datapath, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)


def readjson(datapath):
    """
    Read data from a JSON file.

    Args:
        datapath: Path to the JSON file

    Returns:
        The parsed JSON data
    """
    logger.info(f"Reading file at {datapath}")
    with open(datapath, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


class CheckpointManager:
    """A class to manage checkpoints for data processing tasks"""

    def __init__(self, cache_dir, task_name):
        self.cache_dir = cache_dir
        self.task_name = task_name
        self.checkpoint_file = os.path.join(cache_dir, f"{task_name}_checkpoint.json")
        os.makedirs(cache_dir, exist_ok=True)

    def save_checkpoint(self, data, current_index, total_items):
        """Save current progress and data to checkpoint"""
        checkpoint = {
            "data": data,
            "current_index": current_index,
            "total_items": total_items,
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        }
        writejson(checkpoint, self.checkpoint_file)
        logger.info(f"Checkpoint saved at index {current_index}/{total_items}")

    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            checkpoint = readjson(self.checkpoint_file)
            logger.info(
                f"Resuming from checkpoint at index {checkpoint['current_index']}/{checkpoint['total_items']}"
            )
            return checkpoint
        return None

    def clear_checkpoint(self):
        """Clear existing checkpoint"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint cleared")

    def process_with_checkpoint(self, input_data, process_func, batch_size=1):
        """Process data with checkpoint support

        Args:
            input_data: List of items to process
            process_func: Function that processes a batch of items
            batch_size: Number of items to process in each batch

        Returns:
            List of processed items

        Raises:
            Exception: If processing fails
        """
        checkpoint = self.load_checkpoint()

        if checkpoint:
            # Resume from checkpoint
            processed_data = checkpoint["data"]
            start_idx = checkpoint["current_index"]
        else:
            # Start fresh
            processed_data = []
            start_idx = 0

        total_items = len(input_data)

        try:
            for idx in range(start_idx, total_items, batch_size):
                print(
                    f"{Fore.BLUE}--- Processing Batch {idx}/{total_items} ---{Style.RESET_ALL}"
                )
                batch = input_data[idx : min(idx + batch_size, total_items)]
                batch_results = process_func(batch)
                processed_data.extend(batch_results)

                # Save checkpoint after each batch
                self.save_checkpoint(processed_data, idx + len(batch), total_items)

            # Clear checkpoint after successful completion
            self.clear_checkpoint()
            return processed_data

        except Exception as e:
            # Save checkpoint on error
            self.save_checkpoint(processed_data, idx, total_items)
            logger.error(f"Error during processing: {str(e)}")
            raise e


class MultiProcessHandler:
    """
    Handles parallel processing of tasks using multiple processes.

    This class manages the parallel execution of a given function across
    multiple processes, with progress tracking.

    Attributes:
        process_item_func (Callable): Function to process individual items
        max_workers (int): Maximum number of parallel processes
    """

    def __init__(self, process_item_func, max_workers=4, theading=True):
        self.process_item_func = process_item_func
        self.max_workers = max_workers
        self.theading = theading

    def run(self, items):
        """
        Run parallel processing on items.

        Args:
            items: List of argument tuples for process_item_func

        Returns:
            List of results in the same order as input items
        """
        results = [None] * len(items)
        # 初始化进度条（建议用 with 上下文管理器自动管理资源）
        if self.theading:  # works for io bottleneck task
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # 提交所有任务，并记录每个 Future 对应的索引位置
                future_to_index = {
                    executor.submit(self.process_item_func, *args): idx
                    for idx, args in enumerate(items)
                }

                # 使用tqdm创建进度条
                # with Progress(
                #     SpinnerColumn(),  # 显示一个动态的加载动画
                #     TextColumn("[bold blue]{task.description}"),  # 任务描述，加粗蓝色
                #     BarColumn(bar_width=40, complete_style="green", finished_style="green", pulse_style="yellow"),  # 进度条
                #     TaskProgressColumn(),  # 显示百分比进度
                #     TextColumn("({task.completed}/{task.total})", justify="right"),  # 当前项/总项数
                #     TimeRemainingColumn(),  # 显示剩余时间 (ETA)
                #     transient=True,  # 完成后自动清除
                # ) as progress:
                with tqdm(
                    total=len(future_to_index),
                    desc=f"Processing {self.process_item_func.__name__}",
                    mininterval=5.0,
                    maxinterval=10.0,
                    leave=True,
                ) as pbar:
                    # task = progress.add_task(f"Processing [bold cyan]{self.process_item_func.__name__}[/bold cyan]", total=len(items))

                    # 逐个等待并获取结果，将结果放回对应的索引位置
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        results[idx] = future.result()
                        # progress.update(task,advance=1)  # 更新进度条
                        pbar.update(1)
                        # logger.info(f'{idx} done')

        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # 提交所有任务，并记录每个 Future 对应的索引位置
                future_to_index = {
                    executor.submit(self.process_item_func, *args): idx
                    for idx, args in enumerate(items)
                }

                # 使用tqdm创建进度条
                # with Progress(
                #     SpinnerColumn(),  # 显示一个动态的加载动画
                #     TextColumn("[bold blue]{task.description}"),  # 任务描述，加粗蓝色
                #     BarColumn(bar_width=40, complete_style="green", finished_style="green", pulse_style="yellow"),  # 进度条
                #     TaskProgressColumn(),  # 显示百分比进度
                #     TextColumn("({task.completed}/{task.total})", justify="right"),  # 当前项/总项数
                #     TimeRemainingColumn(),  # 显示剩余时间 (ETA)
                #     transient=True,  # 完成后自动清除
                # ) as progress:
                with tqdm(
                    total=len(future_to_index),
                    desc=f"Processing {self.process_item_func.__name__}",
                    mininterval=5.0,
                    maxinterval=10.0,
                    leave=True,
                ) as pbar:
                    # task = progress.add_task(f"Processing [bold cyan]{self.process_item_func.__name__}[/bold cyan]", total=len(items))

                    # 逐个等待并获取结果，将结果放回对应的索引位置
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        results[idx] = future.result()
                        # progress.update(task,advance=1)  # 更新进度条
                        pbar.update(1)
                        # logger.info(f'{idx} done')
        return results


# 为了参数是 list 当中每一项, 防止项被拆解, 需要保证完整性
def preprocess_list(input_list: list, additional_args: list = []):
    """
    Prepare input list for parallel processing by ensuring each item remains intact.

    Args:
        input_list: List of items to process
        additional_args: Additional arguments to append to each item

    Returns:
        List of lists, each containing an item and any additional arguments
    """
    list_of_list_item = []
    for item in input_list:
        list_item = [item]
        list_item.extend(additional_args)
        list_of_list_item.append(list_item)
    return list_of_list_item


def sample_process_func(a: int, b: int, c: int = 1) -> int:
    """
    Example function for testing parallel processing.

    Args:
        a: First number
        b: Second number
        c: Optional third number (default: 1)

    Returns:
        Sum of a, b, and c
    """
    time.sleep(2)
    return a + b + c


def parallel_wrapper(
    input_list,
    process_item_func,
    cache_dir="./",
    cache_file="cache_tmp",
    max_workers=1,
    savebatch=1000,
):
    """
    High-level wrapper for parallel processing with checkpoint support.

    Args:
        input_list: List of items to process
        process_item_func: Function to process items
        cache_dir: Directory for checkpoint files
        cache_file: Base name for checkpoint files
        max_workers: Maximum number of parallel processes
        savebatch: Number of items to process before saving checkpoint

    Returns:
        List of processed items

    For example：[(1,2,3), ...] -> process_item_func(1,2,3)
    """
    # Initialize checkpoint manager
    cache_file = cache_file + "_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    checkpoint_manager = CheckpointManager(
        cache_dir,
        cache_file,
    )

    def process_batch(batch):
        """Process a batch of items with roll out and judge"""
        multiprocess_handler = MultiProcessHandler(
            process_item_func, max_workers=max_workers
        )
        return multiprocess_handler.run(batch)

    # Process data with checkpoint support
    return checkpoint_manager.process_with_checkpoint(
        input_list, process_batch, batch_size=savebatch
    )


# test
def test1():
    # 初始化 MultiProcessHandler，设置要执行的函数和并行进程数量
    handler = MultiProcessHandler(process_item_func=sample_process_func, max_workers=3)

    # 构造测试数据，每个元素都是 (a, b, c=可选)
    # 如果 c 不传，就会使用默认值 c=1
    test_items = [
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
    ]

    # 并行处理并获取结果
    results = handler.run(test_items)

    # 打印结果
    print("输入:", test_items)
    print("输出: a + b + c =", results)


# def test_bar():
#     # 创建一个全局的 Progress 对象
#     progress = Progress(
#         SpinnerColumn(),  # 动态加载动画
#         TextColumn("[bold blue]{task.description}"),  # 任务描述
#         BarColumn(bar_width=40, complete_style="green", finished_style="green", pulse_style="yellow"),  # 进度条
#         TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # 完成百分比
#         TextColumn("({task.completed}/{task.total})", justify="right"),  # 当前项/总项数
#         TimeRemainingColumn(),  # 剩余时间 (ETA)
#     )

#     # 总项数
#     total_items = 100

#     # 启动进度条
#     progress.start()
#     task = progress.add_task("[cyan]Processing...", total=total_items)

#     # 模拟处理逻辑
#     def process_batch(batch_size):
#         for _ in range(batch_size):
#             # 模拟处理一项

#             time.sleep(0.1)
#             progress.update(task, advance=1)

#     # 在不同位置更新进度条
#     try:
#         # 第一批处理
#         process_batch(30)

#         # 模拟一些其他操作
#         print("\nPerforming intermediate operations...\n")

#         # 第二批处理
#         process_batch(50)
#         print("\nPerforming intermediate operations...\n")
#         # 最后一批处理
#         process_batch(20)
#     finally:
#         # 确保进度条正确关闭
#         progress.stop()


def test2():

    # 构造测试数据，每个元素都是 (a, b, c=可选)
    # 如果 c 不传，就会使用默认值 c=1
    test_items = [
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
        (1, 2),  # a=1, b=2, c=1(默认) -> 1 + 2 + 1 = 4
        (3, 4, 10),  # a=3, b=4, c=10   -> 3 + 4 + 10 = 17
        (5, 6),  # a=5, b=6, c=1(默认) -> 12
        (7, 8, 2),  # a=7, b=8, c=2    -> 17
    ]

    results = parallel_wrapper(
        test_items,
        sample_process_func,
        cache_dir="./",
        cache_file="test_cache",
        max_workers=3,
    )
    # 打印结果
    print("输入:", test_items)
    print("输出: a + b + c =", results)


if __name__ == "__main__":
    test2()
    # test_bar()
