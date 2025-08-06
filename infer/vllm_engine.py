from typing import Any, Dict, List, Optional

from tqdm import tqdm
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def build_vllm_engine(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    engine_args: Optional[Dict[str, Any]] = None,
):
    """
    构建vLLM引擎

    Args:
        model_name: 模型名称
        engine_args: 引擎参数字典，支持以下关键参数：
            - tensor_parallel_size: int, 模型并行度
            - dtype: str, 计算精度 (float16, bfloat16等)
            - enable_prefix_caching: bool, 是否启用前缀缓存
            - seed: int, 随机种子
            - gpu_memory_utilization: float, GPU显存使用率
            - trust_remote_code: bool, 是否信任远程代码
            - device: str, 指定GPU设备 (如 "cuda:0")
    """

    if engine_args is None:
        engine_args = {}

    parser = FlexibleArgumentParser()
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)

    # 设置基础默认值
    default_args = {"model": model_name, "trust_remote_code": True}

    # 合并用户提供的参数
    if engine_args:
        default_args.update(engine_args)

    # 使用set_defaults正确设置所有参数
    engine_group.set_defaults(**default_args)

    # 使用空列表解析参数，避免读取命令行参数
    args = parser.parse_args([])
    args_dict = vars(args)

    llm = LLM(**args_dict)

    return llm


def vllm_inference(
    llm,
    sampling_params,
    conversations: List[List[Dict[str, str]]],
    batch_size: int = 32,
    chat_template: Optional[str] = None,
    use_tqdm: bool = False,
):
    # 参数验证
    if not conversations:
        raise ValueError("Conversations list cannot be empty")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not all(isinstance(conv, list) for conv in conversations):
        raise ValueError("Each conversation must be a list of messages")

    all_responses = []
    try:
        for i in tqdm(
            range(0, len(conversations), batch_size), desc=f"BATCHSIZE {batch_size}"
        ):
            batch = conversations[i : i + batch_size]

            try:
                if chat_template is not None:
                    outputs = llm.chat(
                        batch,
                        sampling_params,
                        chat_template=chat_template,
                        use_tqdm=use_tqdm,
                    )
                else:
                    outputs = llm.chat(batch, sampling_params, use_tqdm=use_tqdm)

                responses = [output.outputs[0].text for output in outputs]
                all_responses.extend(responses)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                # 可以选择跳过这个批次继续处理，或者返回已处理的结果
                continue

        return all_responses
    except Exception as e:
        print(f"Fatal error during inference: {str(e)}")
        return all_responses
