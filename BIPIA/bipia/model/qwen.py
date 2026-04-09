# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from transformers import AutoTokenizer
from vllm import SamplingParams

from .vllm_worker import vLLMModel

__all__ = ["QwenModel", "Qwen3", "Qwen3WOSystem"]

class QwenModel(vLLMModel):
    def load_tokenizer(self):
        # 官方推荐 transformers >= 4.51.0，通常不需要再强制 trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            trust_remote_code=self.config.get("trust_remote_code", True),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        """
        重写 vLLMModel 的配置。
        Qwen3 官方文档强烈警告：DO NOT use greedy decoding (temperature=0)!
        因此我们将其修改为官方推荐的参数。
        """
        conv_template = self.get_conv_template()
        max_new_tokens = self.kwargs.get("max_new_tokens", 4096) # Qwen3 输出可能较长
        
        # 读取 yaml 配置中是否启用了 thinking mode (默认设为 False，以对齐传统 Instruct 模型)
        self.enable_thinking = self.config.get("enable_thinking", False)
        
        if self.enable_thinking:
            temperature = 0.6
            top_p = 0.95
        else:
            temperature = 0.7
            top_p = 0.8
            
        self.generation_config = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            max_tokens=max_new_tokens,
            stop=conv_template.stop_str,
            stop_token_ids=conv_template.stop_token_ids,
        )
        return self.generation_config

    def post_process(self, responses: list[str]):
        """
        处理 Qwen3 独有的 <think>...</think> 标签。
        在 BIPIA 评估中，我们通常只需要模型最终的 answer 来判断其是否抵御了 Prompt Injection，
        所以我们需要剥离推导过程。
        """
        processed_responses = []
        for response in responses:
            # 去掉可能的特殊 token 残留
            response = response.strip()
            
            # 使用正则过滤掉 <think> 到 </think> 之间的所有内容 (包含换行符)
            if "<think>" in response and "</think>" in response:
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            # 兼容截断情况：如果有 <think> 但没有 </think>（生成超出 max_tokens）
            elif "<think>" in response:
                # 只保留 think 之前的内容（通常为空），或者直接返回空字符串
                response = response.split("<think>")[0].strip()
                
            processed_responses.append(response)
            
        return processed_responses


class Qwen3(QwenModel):
    """带 System Prompt 的常规对话/评估模式"""
    require_system_prompt = True


class Qwen3WOSystem(QwenModel):
    """不带 System Prompt，直接注入纯 Prompt 模式"""
    require_system_prompt = False