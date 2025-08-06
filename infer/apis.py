"""
API interface module for LLM model inference.
This module provides functions to interact with LLM APIs with load balancing
and retry mechanisms.
"""

import argparse
import copy
import json
import os
import sys
import time
from itertools import cycle
from typing import Any, Dict, List, Optional

from openai import OpenAI

from global_config import PROJECT_ROOT, URL_LIST
from logs.logger import logger
from modules.utils import (
    is_en_lang,
    readjson,
    readjsonl,
    seed_everything,
    writejson,
    writejsonl,
)

# Create cyclic iterators for each URL group for load balancing
list_url_cycle = [cycle(url_group) for url_group in URL_LIST]


class APIError(Exception):
    """Custom exception for API-related errors."""

    pass


def get_res_cycle(
    user_prompt: str,
    model_name: str = "qwen",
    temperate: float = 0.0,
    max_tokens: int = 1024,
    url_level: int = 0,
    max_retry: int = 1,
    return_extra_info: bool = False,
) -> Optional[str]:
    """
    Get response from LLM API with automatic endpoint cycling and load balancing.

    This function cycles through available endpoints at the specified URL level,
    implementing basic load balancing. When an endpoint fails, it retries with
    the next endpoint up to MAX_API_RETRY times.

    Args:
        user_prompt (str): The prompt to send to the model
        model_name (str): Name of the model to use
        temperate (float): Temperature parameter for response generation
        max_tokens (int): Maximum number of tokens in the response
        url_level (int): Level of model to use (higher means more capable model)

    Returns:
        Optional[str]: Model's response or None if all retries fail

    Raises:
        APIError: If url_level is invalid
    """
    MAX_API_RETRY = max_retry

    if url_level >= len(list_url_cycle):
        url_level = len(list_url_cycle) - 1
        logger.warning(
            f"URL level {url_level} is too high, using highest available level: {url_level}"
        )

    for attempt in range(MAX_API_RETRY):
        current_endpoint = next(list_url_cycle[url_level])

        try:
            client = OpenAI(
                api_key="token-abc123",  # TODO: Make this configurable
                base_url=current_endpoint,
            )

            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperate,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
            usage = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.choices[0].finish_reason != "stop":
                raise ValueError("LENGTH: response is too long and cut")
            if return_extra_info:
                return response.choices[0].message.content, [usage]
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.warning(
                f"Endpoint {current_endpoint} failed (attempt {attempt + 1}/{MAX_API_RETRY}). "
                f"Error: {str(e)}"
            )
            time.sleep(min(5 * (attempt + 1), 10))  # Exponential backoff with cap

    logger.error(f"All {MAX_API_RETRY} attempts failed for URL level {url_level}")
    return None


def get_res(
    user_prompt: str,
    model_name: str = "qwen",
    temperature: float = 0.0,
    api_endpoint: str = "http://127.0.0.1:8007/v1",
    max_tokens: int = 1024,
    api_key: str = "token-abc123",
    receive_length: bool = False,
) -> Optional[str]:
    """
    Get response from LLM API using a specific endpoint.

    This function attempts to get a response from a specific API endpoint,
    with retry capability on failure.

    Args:
        user_prompt (str): The prompt to send to the model
        model_name (str): Name of the model to use
        temperate (float): Temperature parameter for response generation
        api_endpoint (str): Specific API endpoint to use
        max_tokens (int): Maximum number of tokens in the response

    Returns:
        Optional[str]: Model's response or None if all retries fail
    """
    MAX_API_RETRY = 2
    flag_length = False
    for attempt in range(MAX_API_RETRY):
        try:
            client = OpenAI(
                api_key=api_key, base_url=api_endpoint  # TODO: Make this configurable
            )

            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
            if not receive_length and response.choices[0].finish_reason != "stop":
                # logger.warning(f"LENGTH: response is too long and cut")
                with open("badcase.txt", "a") as file:
                    file.write(
                        f"\n\n--------badcase--------\n\n{user_prompt}\n\n{response.choices[0].message.content}\n\n"
                    )
                continue
            return response.choices[0].message.content

        except Exception as e:
            logger.warning(
                f"Endpoint {api_endpoint} failed (attempt {attempt + 1}/{MAX_API_RETRY}). "
                f"Error: {str(e)}"
            )
            time.sleep(min(5 * (attempt + 1), 5))  # Exponential backoff with cap

    logger.error(f"All {MAX_API_RETRY} attempts failed for endpoint {api_endpoint}")
    return None


test_query1 = """Write code for a Discord bot using Discord.js v14. The bot has one command, ban. All commands are slash commands. The bot operates in a server with over 1000 members. Include error handling for cases where the user does not have permission to ban members.Please reply in English and capitalize all your words. The response must contain at least 3 placeholders (e.g., [restaurant]). Your response should contain at least 3 sentences.. Include exactly 3 bullet points in your response. The bullet points should be in the form of:\n* This is bullet 1\n* This is bullet 2\n...\n. """

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="roll out and judge")
    # general para
    parser.add_argument("--api_endpoint", type=str, default="http://127.0.0.1:8007/v1")
    parser.add_argument("--api_key", type=str, default="token-abc123")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    query = "Hello, how are you?"
    # query = test_query1

    response = get_res(
        query,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if response:
        print("Response:", response)
    else:
        print("Failed to get response from any endpoint")
