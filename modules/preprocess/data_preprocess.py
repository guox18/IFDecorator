import argparse
import copy
import json
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

from langdetect import detect
from tqdm import tqdm

from global_config import PROJECT_ROOT

##
from logs.logger import logger
from modules.utils import (
    is_en_lang,
    readjson,
    readjsonl,
    seed_everything,
    writejson,
    writejsonl,
)


class BaseDataProcessor(ABC):
    """Base class for all data processors"""

    def __init__(self, input_path: str, sample_size: int = None):
        self.input_path = input_path

        self.sample_size = sample_size
        self.source = os.path.basename(input_path).split(".")[0]

    def read_data(self) -> List[Dict]:
        """Read data from input file"""
        if self.input_path.endswith(".jsonl"):
            return readjsonl(self.input_path)
        else:
            data = readjson(self.input_path)
            if isinstance(data, dict):
                return data.get("data", []) if "data" in data else [data]
            return data

    @abstractmethod
    def process_item(self, item: Dict) -> Dict:
        """Process a single item from the dataset"""
        pass

    def is_english(self, text: str) -> bool:
        try:
            if "en" in detect(text):
                return True
            return False
        except:
            return False

    def process(self):
        """Process the entire dataset"""
        logger.info(f"Processing {self.source}...")
        data = self.read_data()
        # for efficiency
        random.shuffle(data)
        data = data[: self.sample_size * 3]

        processed_data = []

        for item in tqdm(data, desc=f"Processing {self.source} items"):
            processed = self.process_item(item)
            if processed and "prompt" in processed and "response" in processed:
                # Check if both prompt and response are in English
                if self.is_english(processed["prompt"]) and self.is_english(
                    processed["response"]
                ):
                    processed_data.append(processed)

        if self.sample_size and len(processed_data) < self.sample_size:
            logger.warning(f"self.source: too many instructions filtered")
            self.sample_size = len(processed_data)
        processed_data = random.choices(processed_data, k=self.sample_size)
        logger.info(f"self.source: {self.sample_size}")
        return processed_data


class OpenHermesProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "conversations" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["conversations"]

        # 只取第一轮对话
        human_msg = next((msg for msg in messages if msg.get("from") == "human"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("from") == "gpt"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("value", "")
            result["response"] = assistant_msg.get("value", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = item.get("category", None)
            return result
        return None


class ShareGPTProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "conversations" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["conversations"]

        # # ShareGPT可能包含额外的元数据
        # if "metadata" in item:
        #     result["metadata"] = item["metadata"]

        human_msg = next((msg for msg in messages if msg.get("from") == "human"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("from") == "gpt"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("value", "")
            result["response"] = assistant_msg.get("value", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class AlpacaProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "messages" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["messages"]

        human_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("content", "")
            result["response"] = assistant_msg.get("content", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class WizardLMProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "messages" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["messages"]

        human_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("content", "")
            result["response"] = assistant_msg.get("content", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class OrcaChatProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "messages" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["messages"]

        human_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("content", "")
            result["response"] = assistant_msg.get("content", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class Oasst2Processor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "messages" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["messages"]

        human_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("content", "")
            result["response"] = assistant_msg.get("content", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class NoRobotsProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        if "messages" not in item:
            return None

        result = {"source": item["source"] if "source" in item else self.source}
        messages = item["messages"]

        human_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg.get("role") == "assistant"), None
        )

        if human_msg and assistant_msg:
            result["prompt"] = human_msg.get("content", "")
            result["response"] = assistant_msg.get("content", "")
            result["id"] = (
                f"{self.source}_{hash(result['prompt'] + result['response'])}"
            )
            result["original_category"] = (
                item["category"] if "category" in item else None
            )
            return result
        return None


class SupernaturalProcessor(BaseDataProcessor):
    def process_item(self, item: Dict) -> Dict:
        hash_id = hash(item.get("prompt", "") + item.get("response", ""))
        return {
            "source": self.source,
            "prompt": item.get("prompt", ""),
            "response": item.get("response", ""),
            "id": f"{self.source}_{hash_id}",
            "original_category": item["category"] if "category" in item else None,
        }


def get_processor_class(source: str) -> type:
    """Get the appropriate processor class based on the source"""
    processors = {
        "openhermes2_5": OpenHermesProcessor,
        "supernatural": SupernaturalProcessor,
        "sharegpt_clean_en_reduce_rep": ShareGPTProcessor,
        "wizardLM": WizardLMProcessor,
        "orca_chat": OrcaChatProcessor,
        "alpaca": AlpacaProcessor,
        "oasst2": Oasst2Processor,
        "no_robots": NoRobotsProcessor,
    }
    return processors.get(source, BaseDataProcessor)


def process_dataset(source: str, input_dir: str, sample_size: int = None):
    """Process a single dataset"""
    # Try json first, if not found try jsonl
    json_path = os.path.join(input_dir, f"{source}.json")
    jsonl_path = os.path.join(input_dir, f"{source}.jsonl")
    input_path = jsonl_path if os.path.exists(jsonl_path) else json_path

    processor_class = get_processor_class(source)
    processor = processor_class(input_path, sample_size)
    return processor.process()


### 500k


def process_all_datasets(input_dir: str = None, output_path: str = None):
    """Process all datasets with their respective sample sizes"""
    if input_dir is None:
        input_dir = os.path.join(PROJECT_ROOT, "data/v0_raw")
    
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data/v1_seed/random_selected_data_350k.jsonl")
    
    sample_sizes = {
        "sharegpt_clean_en_reduce_rep": 51000,  # 94145
        "no_robots": 19000,  # 19000
        "openhermes2_5": 200000,  # 1001551
        "supernatural": 10000,  # 1990915
        "wizardLM": 25000,  # 46034
        "orca_chat": 25000,  # 44885
        "alpaca": 15000,  # 15304
        "oasst2": 5000,  # 5236
    }

    all_list = []

    for source, sample_size in tqdm(sample_sizes.items(), desc="Processing datasets"):
        ret_list = process_dataset(source, input_dir, sample_size)
        all_list.extend(ret_list)  # Fixed bug: was extending all_list with itself
    writejsonl(
        all_list,
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Process datasets for training")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/v0_raw"),
        help="Input directory containing the raw data files"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/v1_seed/random_selected_data_350k.jsonl"),
        help="Output path for the processed data"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Process all datasets
    process_all_datasets(args.input_dir, args.output_path)
    logger.info(f"Processing completed. Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
