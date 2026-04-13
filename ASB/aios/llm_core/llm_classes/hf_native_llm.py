# Run models from huggingface using the transformers library

import torch
from .constant import MODEL_CLASS
from .base_llm import BaseLLM
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

from pyopenagi.utils.chat_template import Response

from ...utils.utils import get_from_env

import re
import json

class HfNativeLLM(BaseLLM):

    def build_prompt(self, messages, tools):
        if tools:
            messages = self.tool_calling_input_format(messages, tools)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

    def load_llm_and_tokenizer(self) -> None:
        """ fetch the model from huggingface and run it """
        
        # 拦截 max_gpu_memory 为 None 或空字典的情况
        if self.max_gpu_memory:
            self.max_gpu_memory = self.convert_map(self.max_gpu_memory)
        else:
            self.max_gpu_memory = None

        self.auth_token = None

        """ only casual lms for now """
        # 使用 AutoModelForCausalLM 直接加载，并强制使用 bfloat16 (Llama-3.1 标准精度)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            max_memory=self.max_gpu_memory,
            use_auth_token=self.auth_token,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.auth_token
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def parse_tool_calls(self, result):
        # Extract all JSON-array candidates, then pick the last one that can be
        # normalized into [{"name": ..., "parameters": {...}}].
        pattern = r'\[\s*\{[\s\S]*?\}\s*\]'
        matches = re.findall(pattern, result, re.DOTALL)
        if not matches:
            return []

        def normalize_calls(candidate):
            if not isinstance(candidate, list):
                return None

            normalized = []
            for item in candidate:
                if not isinstance(item, dict):
                    return None

                # Case 1: already in expected format
                if "name" in item:
                    name = item.get("name")
                    params = item.get("parameters", {})
                    if not isinstance(name, str):
                        return None
                    if params is None:
                        params = {}
                    if not isinstance(params, dict):
                        return None
                    normalized.append({"name": name, "parameters": params})
                    continue

                # Case 2: OpenAI-ish format {"type":"function","function":{"name":...,"arguments":...}}
                fn = item.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    name = fn["name"]
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if args is None:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    normalized.append({"name": name, "parameters": args})
                    continue

                # Not a tool-call item
                return None

            return normalized

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
            except Exception:
                continue
            normalized = normalize_calls(parsed)
            if normalized is not None:
                return normalized

        return []

    def process(self,
                agent_process,
                temperature=0.0) -> None:
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )

        messages = agent_process.query.messages
        tools = agent_process.query.tools
        message_return_type = agent_process.query.message_return_type

        """ context_manager works only with open llms """
        if self.context_manager.check_restoration(agent_process.get_pid()):
            restored_context = self.context_manager.gen_recover(
                agent_process.get_pid()
            )
            start_idx = restored_context["start_idx"]
            beams = restored_context["beams"]
            beam_scores = restored_context["beam_scores"]
            beam_attention_mask = restored_context["beam_attention_mask"]

            outputs = self.generate(
                search_mode = "beam_search",
                beam_size = 1,
                beams = beams,
                beam_scores = beam_scores,
                beam_attention_mask = beam_attention_mask,
                max_new_tokens = self.max_new_tokens,
                start_idx = start_idx,
                timestamp = agent_process.get_time_limit()
            )
        else:
            """ use the system prompt otherwise """

            prompt = self.build_prompt(messages, tools)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

            attention_mask = input_ids != self.tokenizer.pad_token_id
            
            # 使用自适应的模型设备，防止 OOM 或设备不匹配错误
            current_device = self.model.device
            input_ids = input_ids.to(current_device)
            attention_mask = attention_mask.to(current_device)

            outputs = self.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                search_mode = "beam_search",
                beam_size = 1,
                max_new_tokens=self.max_new_tokens,
                start_idx = 0,
                timestamp = agent_process.get_time_limit()
            )
            # TODO temporarily
            outputs["result"] = outputs["result"][input_ids.shape[-1]:]
        # output_ids = outputs
        # print(output_ids)
        output_ids = outputs["result"]

        """ devectorize the output """
        # Llama-3.1 含有特殊系统 token，跳过它们能让输出更干净
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if outputs["finished_flag"]: # finished flag is set as True

            if self.context_manager.check_restoration(
                agent_process.get_pid()):
                self.context_manager.clear_restoration(
                    agent_process.get_pid()
                )

            if tools:
                tool_calls = self.parse_tool_calls(result)
                agent_process.set_response(
                    Response(
                        response_message = result,  # 关键修复 2：把 None 改回 result
                        tool_calls = tool_calls
                    )
                )
            else:
                agent_process.set_response(
                    Response(
                        response_message = result
                    )
                )
            agent_process.set_status("done")

        else:
            """ the module will automatically suspend if reach the time limit """
            self.logger.log(
                f"{agent_process.agent_name} is switched to suspending due to the reach of time limit ({agent_process.get_time_limit()}s).\n",
                level = "suspending"
            )
            self.context_manager.gen_snapshot(
                agent_process.get_pid(),
                context = {
                    "start_idx": outputs["start_idx"],
                    "beams": outputs["beams"],
                    "beam_scores": outputs["beam_scores"],
                    "beam_attention_mask": outputs["beam_attention_mask"]
                }
            )
            if message_return_type == "json":
                result = self.parse_json_format(result)
            agent_process.set_response(
                Response(
                    response_message = result
                )
            )
            agent_process.set_status("suspending")

        agent_process.set_end_time(time.time())

    def generate(self,
                 input_ids: torch.Tensor = None,
                 attention_mask: torch.Tensor = None,
                 beams: torch.Tensor = None,
                 beam_scores: torch.Tensor = None,
                 beam_attention_mask: torch.Tensor = None,
                 beam_size: int = None,
                 max_new_tokens: int = None,
                 search_mode: str = None,
                 start_idx: int = 0,
                 timestamp: int = None
                 ):
        """ only supports beam search generation """
        if search_mode == "beam_search":
            output_ids = self.beam_search(
                input_ids = input_ids,
                attention_mask = attention_mask,
                beam_size = beam_size,
                beams = beams,
                beam_scores = beam_scores,
                beam_attention_mask = beam_attention_mask,
                max_new_tokens = max_new_tokens,
                start_idx = start_idx,
                timestamp = timestamp
            )
            return output_ids
        else:
            # TODO: greedy support
            return NotImplementedError

    def beam_search(self,
                    input_ids: torch.Tensor = None,
                    attention_mask: torch.Tensor = None,
                    beams=None,
                    beam_scores=None,
                    beam_attention_mask=None,
                    beam_size: int = None,
                    max_new_tokens: int = None,
                    start_idx: int = 0,
                    timestamp: int = None
                    ):

        """
        beam search gets multiple token sequences concurrently and calculates
        which token sequence is the most likely opposed to calculating the
        best token greedily
        """
        current_device = self.model.device

        if beams is None or beam_scores is None or beam_attention_mask is None:
            beams = input_ids.repeat(beam_size, 1)
            beam_attention_mask = attention_mask.repeat(beam_size, 1)
            beam_scores = torch.zeros(beam_size, device=current_device)

        start_time = time.time()
        finished_flag = False
        idx = start_idx

        # ================= Llama-3.1 Terminators 设定 =================
        terminators = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)
        # =============================================================

        for step in range(start_idx, max_new_tokens):
            with torch.no_grad():
                # Obtain logits for the last tokens across all beams
                outputs = self.model(beams, attention_mask=beam_attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply softmax to convert logits to probabilities
                next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # Calculate scores for all possible next tokens for each beam
            next_token_scores = beam_scores.unsqueeze(-1) + torch.log(next_token_probs)

            # Flatten to treat the beam and token dimensions as one
            next_token_scores_flat = next_token_scores.view(-1)

            # Select top overall scores to find the next beams
            top_scores, top_indices = torch.topk(next_token_scores_flat, beam_size, sorted=True)

            # Determine the next beams and their corresponding tokens
            beam_indices = top_indices // next_token_probs.size(1)  # Find which beam the top tokens came from
            token_indices = top_indices % next_token_probs.size(1)  # Find which token within the beam was selected

            # Update beams, scores, and attention masks
            beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(-1)], dim=-1)
            beam_attention_mask = torch.cat([beam_attention_mask[beam_indices], torch.ones_like(token_indices).unsqueeze(-1)], dim=-1)
            beam_scores = top_scores

            # Check for stopping criteria
            if timestamp is not None and time.time() - start_time >= timestamp:
                idx = step
                break

            # Check for completion (适配 Llama-3.1 多终止符)
            last_tokens = beams[:, -1]
            is_finished_mask = torch.zeros_like(last_tokens, dtype=torch.bool)
            for term_id in terminators:
                is_finished_mask |= (last_tokens == term_id)
                
            if torch.all(is_finished_mask):
                idx = step
                finished_flag = True
                break

            if step + 1 == max_new_tokens:
                idx = step
                finished_flag = True
                break

        best_beam_idx = beam_scores.argmax()

        best_beam = beams[best_beam_idx]

        outputs = {
            "finished_flag": finished_flag,
            "start_idx": idx,
            "beams": beams,
            "beam_scores": beam_scores,
            "beam_attention_mask": beam_attention_mask,
            "result": best_beam
        }

        return outputs