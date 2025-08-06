"""
全局配置文件
包含项目的服务器地址、路径配置和模型路径等全局变量
"""

from pathlib import Path

# 服务器配置
SERVER_IP = "127.0.0.1"

# API服务器URL列表
URL_LIST = [
    [
        f"http://{SERVER_IP}:8000/v1",
        f"http://{SERVER_IP}:8001/v1",
        f"http://{SERVER_IP}:8002/v1",
        f"http://{SERVER_IP}:8003/v1",
        f"http://{SERVER_IP}:8004/v1",
        f"http://{SERVER_IP}:8005/v1",
        f"http://{SERVER_IP}:8006/v1",
        f"http://{SERVER_IP}:8007/v1",
    ],
    [
        f"http://{SERVER_IP}:8000/v1",
        f"http://{SERVER_IP}:8001/v1",
    ],
]

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.absolute()
PROJECT_DATA = "data/ifdecorator/pipeline2"

# 模型路径配置
SENTENCE_BERT_PATH = (
    ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)

MODEL_7B_PATH = (
    "hf_hub/models/custom--Qwen2.5-7B-Instruct-tokenizer-modified/"
    "snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75_no_yarn"
)
