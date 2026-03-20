from .base_llm import BaseLLM
import time

# could be dynamically imported similar to other models

from pyopenagi.utils.chat_template import Response
from ...utils.utils import get_from_env

from transformers import AutoTokenizer

from openai import OpenAI
import os
from huggingface_hub import login
#login(token="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")


class vLLM(BaseLLM):
    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console"):
        super().__init__(llm_name, max_gpu_memory, eval_device, max_new_tokens, log_mode)
        
        # 初始化 OpenAI 客户端指向本地 vLLM 端口
        这个需要设置一下，因为上传通不过检验
        '''
        self.client = OpenAI(
            base_url="http://localhost:8000/v1", # 指向本地 vLLM
            api_key="EMPTY" # vLLM 本地默认不需要 Key
        )
        self.model_name = llm_name # 确保这里跟启动 vLLM 的名字一致，或者用 "tgi" 等占位
'''
    def load_llm_and_tokenizer(self) -> None:
        # 客户端模式不需要加载笨重的模型权重
        pass

    def process(self, agent_process, temperature=0.0) -> None:
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        
        messages = agent_process.query.messages
        # 这里的 tools 逻辑和你 Code 2 的配合：
        # 如果 Code 2 传了 tools=None (手动挡)，这里 tools 就是 None
        tools = agent_process.query.tools 

        try:
            # 发送请求给 vLLM Server
            # Server 会自动处理并发 Batching，这才是速度提升的关键！
            if tools:
                # 如果是原生 Tool Calling (Code 2如果没改)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=self.max_new_tokens
                )
            else:
                # 如果是 Code 2 的手动挡 (JSON Injection)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_new_tokens
                )

            # 获取结果
            result = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls

            print(f"***** vLLM Response: {result[:50]}... *****")

            # 封装回 AgentProcess
            from pyopenagi.utils.chat_template import Response
            
            # 兼容处理：如果是原生 Tool Call 返回
            if tool_calls:
                 agent_process.set_response(Response(response_message=None, tool_calls=tool_calls))
            else:
                 agent_process.set_response(Response(response_message=result))

        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            agent_process.set_response(Response(response_message=f"Error: {e}"))

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())