# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict
import yaml
import time
import re
import openai

from accelerate.logging import get_logger

# ----------- 新版错误类型 -----------
from openai import (
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,      # 替代 ServiceUnavailableError
    AuthenticationError,
)

from .base import BaseEval

logger = get_logger(__name__)


def get_retry_time(err_info: str) -> int:
    """从错误信息里提取建议等待时间"""
    z = re.search(r"after (\d+) seconds", err_info)
    return int(z.group(1)) if z else 1


class ModelEval(BaseEval):
    """Compute evaluate metrics with GPT-4"""

    def __init__(self, config: str | dict, judge_fn: Callable, format_fn: Callable):
        super().__init__()
        self.config = self.load_config(config)
        self.judge_fn = judge_fn
        self.format_fn = format_fn

        # ----------- 初始化客户端 -----------
        # 支持 OpenAI / Azure / 自定义 base
        api_key   = self.config.get("api_key")
        api_base  = self.config.get("api_base")
        api_type  = self.config.get("api_type")      # 可选 "azure"
        api_version = self.config.get("api_version")  # Azure 需要

        if api_type == "azure":
            self.client = openai.AzureOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=api_version or "2023-05-15",
            )
        else:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base,
            )

    # ------------------------------------------------------------------
    #  工具函数
    # ------------------------------------------------------------------
    def load_config(self, config: str | dict) -> Dict:
        if isinstance(config, dict):
            return config
        with open(config, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    # ------------------------------------------------------------------
    #  统一重试装饰器（可选，这里保持原风格手写）
    # ------------------------------------------------------------------
    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        """兼容旧接口：循环重试 + 返回 List[str]"""
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.get("model") or self.config.get("engine"),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                # 新版：resp.choices[i].message.content
                return [choice.message.content for choice in resp.choices]

            # ---------------- 可重试异常 ----------------
            except RateLimitError as e:
                logger.debug(e, exc_info=True)
                time.sleep(get_retry_time(str(e)))
            except (APITimeoutError, APIConnectionError) as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except InternalServerError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)

            # ---------------- 不再重试 ----------------
            except (BadRequestError, AuthenticationError) as e:
                logger.warning(e, exc_info=True)
                return []  # 直接返回空列表
            except Exception as e:
                logger.warning(e, exc_info=True)
                return []

    # ------------------------------------------------------------------
    #  Completion 接口（遗留，非 chat）
    # ------------------------------------------------------------------
    def completion(
        self,
        prompt,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    ):
        stop = stop or ["<|im_end|>"]
        while True:
            try:
                resp = self.client.completions.create(
                    model=self.config.get("model") or self.config.get("engine"),
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                return [choice.text for choice in resp.choices]

            # 同 chat_completion 的异常处理
            except RateLimitError as e:
                logger.debug(e, exc_info=True)
                time.sleep(get_retry_time(str(e)))
            except (APITimeoutError, APIConnectionError) as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except InternalServerError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except BadRequestError as e:
                logger.warning(e, exc_info=True)
                return []
            except Exception as e:
                logger.warning(e, exc_info=True)
                return []

    # ------------------------------------------------------------------
    #  评分逻辑
    # ------------------------------------------------------------------
    def _compute_score(self, prediction: str | None = None, **kwargs):
        messages = self.format_fn(prediction, chat=self.config["chat"])
        if self.config["chat"]:
            response = self.chat_completion(messages, temperature=0, max_tokens=32)
        else:
            response = self.completion(messages, temperature=0, max_tokens=32)

        return self.judge_fn(response[0]) if response else -1

    def _batch_compute_score(self, predictions: list[str] | None = None, **kwargs):
        # 非 chat 模式才走批量 completion
        prompts = [
            self.format_fn(p, chat=False) for p in predictions or []
        ]
        responses = self.completion(prompts, temperature=0, max_tokens=32)
        return [self.judge_fn(r) for r in responses] if responses else [-1] * len(prompts)

    def add_batch(self, *, predictions=None, **kwargs):
        if self.config.get("chat", True):
            # chat 模式逐条调用
            batch_asrs = [self._compute_score(prediction=p) for p in predictions or []]
        else:
            batch_asrs = self._batch_compute_score(predictions=predictions)

        self.asrs.extend(batch_asrs)
        return batch_asrs