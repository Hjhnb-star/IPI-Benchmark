import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hf_native_llm import HfNativeLLM


class Qwen3HfLLM(HfNativeLLM):
    """Qwen3-specific HuggingFace runner with safer chat-template defaults."""

    def load_llm_and_tokenizer(self) -> None:
        if self.max_gpu_memory:
            self.max_gpu_memory = self.convert_map(self.max_gpu_memory)
        else:
            self.max_gpu_memory = None

        self.auth_token = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            max_memory=self.max_gpu_memory,
            use_auth_token=self.auth_token,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.auth_token,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def build_prompt(self, messages, tools):
        if tools:
            messages = self.tool_calling_input_format(messages, tools)

        # Keep current behavior by default while allowing explicit switching.
        enable_thinking = self.enable_thinking
        if enable_thinking is None:
            enable_thinking = False

        # Qwen3 chat templates generally require generation prompt for stable output.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )