# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Any, Callable, Tuple

import os
import re
import time
from openai import OpenAI
from openai import (
    # 新版错误类型，直接从 openai 导入
    RateLimitError,
    APIError,
    APIConnectionError,
    APITimeoutError,  # 替代旧版 Timeout
    BadRequestError, # 替代旧版 InvalidRequestError
    # InternalServerError, # 替代旧版 ServiceUnavailableError
    # 其他可能用到的错误，如 AuthenticationError 等，这里保留原始代码捕获的名称
)


from accelerate.logging import get_logger

from .base import BaseModel

__all__ = ["GPTModel", "GPT35", "GPT4"]

logger = get_logger(__name__)

# --- 新增的全局变量用于存储客户端实例 ---
# 推荐在 __init__ 中实例化，但为了兼容性，如果配置相同，可以使用单例。
# 我们在 GPTModel.__init__ 中实例化 client。
# ------------------------------------

def get_retry_time(err_info):
    """
    提取重试时间。
    """
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1


class GPTModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        config = self.load_config(config)
        self.config = config
        
        # ** v1.x 关键修改：初始化 OpenAI 客户端 **
        self.client = OpenAI(
            api_key=self.config.get("api_key", os.environ.get("OPENAI_API_KEY")),
            base_url=self.config.get("api_base", None), # 新版使用 base_url
            # 注意：v1.x 中不再直接支持 api_type/api_version/engine 参数，
            # 如果是 Azure 用户，需在 base_url 中配置 Azure Endpoint
        )


    def chat_completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        success = False
        while not success:
            try:
                # ** v1.x 关键修改：使用 client.chat.completions.create **
                response = self.client.chat.completions.create(
                    model=self.config.get("model", None),
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                success = True
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except APITimeoutError as e: # 替换旧版 Timeout
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e: # 捕获所有通用 API 错误，包括 ServiceUnavailableError
                logger.debug(f"API Error: {e}", exc_info=True)
                time.sleep(1)
            except BadRequestError as e: # 替换旧版 InvalidRequestError
                logger.warning(e, exc_info=True)
                success = True
                response = type('obj', (object,), {'choices': []}) # 模拟返回空列表
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = type('obj', (object,), {'choices': []}) # 模拟返回空列表
        
        try:
            # ** v1.x 关键修改：访问 response 对象的属性 **
            rslts = [i.message.content for i in response.choices]
        except Exception as e:
            # logger.warning(e, exc_info=True)
            rslts = []

        return rslts


    def completion(
        self,
        messages,
        temperature=None,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"],
    ):
        """
        注意：在 v1.x 中，旧的 Completion API (text-davinci-003) 已被弃用或不推荐。
        如果你的模型是 GPT-3.5/4，你应该使用 chat_completion。
        这里我们假设你需要兼容旧的 Completion 模型。
        """
        success = False
        while not success:
            try:
                # ** v1.x 关键修改：使用 client.completions.create **
                response = self.client.completions.create(
                    model=self.config.get("model", None),
                    prompt=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                success = True
            except RateLimitError as e:
                # logger.warning(e, exc_info=True)
                retry_time = get_retry_time(str(e))
                time.sleep(retry_time)
            except APITimeoutError as e: # 替换旧版 Timeout
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except BadRequestError as e: # 替换旧版 InvalidRequestError
                logger.warning(e, exc_info=True)
                success = True
                response = type('obj', (object,), {'choices': []}) # 模拟返回空列表
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True
                response = type('obj', (object,), {'choices': []}) # 模拟返回空列表

        # ** v1.x 关键修改：访问 response 对象的属性 **
        rslts = [i.text for i in response.choices]
        return rslts


    def generate(self, data: Any, **kwargs):
        temperature = kwargs.pop("temperature", 0)
        
        # 从配置中获取模型名称，判断是否需要使用 Completion API
        model_name = self.config.get("model", "")
        # 如果模型名称是 text-xxx 等 Completion 模型，则使用 completion 方法
        is_completion_model = model_name.startswith("text-") or model_name.startswith("code-")
        
        # 兼容旧代码逻辑，如果 config 中 chat 为 True 则使用 chat_completion
        if self.config.get("chat", not is_completion_model):
            rslts = []
            for message in data["message"]:
                rslt = self.chat_completion(message, temperature=temperature)
                rslts.extend(rslt)
        else:
            rslts = self.completion(data["message"], temperature=temperature)
        return rslts


class GPTModelWSystem(GPTModel):
    require_system_prompt = True

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        system_prompt, user_prompt = prompt_construct_fn(example)

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35(GPTModelWSystem):
    pass


class GPT4(GPTModelWSystem):
    pass


class GPTModelWOSystem(GPTModel):
    require_system_prompt = False

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        # prompt_construct_fn 应该只返回 user_prompt
        user_prompt = prompt_construct_fn(example) 
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."

        if self.config["chat"]:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            example["message"] = message
        else:
            system_message = "<|im_start|>system\n{}\n<|im_end|>".format(system_prompt)
            user_message = "\n<|im_start|>{}\n{}\n<|im_end|>".format(
                "user", user_prompt
            )

            message = system_message + user_message + "\n<|im_start|>assistant\n"
            example["message"] = message
        return example


class GPT35WOSystem(GPTModelWOSystem):
    pass


class GPT4WOSystem(GPTModelWOSystem):
    pass