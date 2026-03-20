import os
import json
import re  # 必须导入
import time
from threading import Thread
import importlib

# 引入项目内部依赖
from .agent_process import AgentProcess
from ..utils.logger import AgentLogger
from ..utils.chat_template import Query
from ..queues.llm_request_queue import LLMRequestQueue
from pyopenagi.tools.simulated_tool import SimulatedTool

class CustomizedThread(Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

    def join(self):
        super().join()
        return self.result

class BaseAgent:
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str
        ):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.tool_names = self.config["tools"]

        self.agent_process_factory = agent_process_factory

        self.tool_list = dict()
        self.tools = []
        # self.load_tools(self.tool_names)

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual" # (mannual, automatic)
        self.rounds = 0

        self.log_mode = log_mode
        self.logger = self.setup_logger()

        self.set_status("active")
        self.set_created_time(time.time())


    def run(self):
        '''Execute each step to finish the task.'''
        pass

    def build_system_instruction(self):
        pass

    # ================= [核心修改] 极度鲁棒的 workflow 提取函数 =================
    def _validate_and_clean_workflow(self, workflow):
        """
        辅助函数：验证并清洗 workflow，确保是合法的 list[dict]
        """
        if not workflow:
            return None

        if not isinstance(workflow, list):
            workflow = [workflow]

        valid_steps = []
        for step in workflow:
            if not isinstance(step, dict):
                continue
            if "message" not in step or "tool_use" not in step:
                continue
            # 确保 tool_use 是 list，防止模型输出 null 或 单个字符串
            if not isinstance(step["tool_use"], list):
                if step["tool_use"] is None:
                    step["tool_use"] = []
                else:
                    step["tool_use"] = [str(step["tool_use"])]
            
            valid_steps.append(step)

        return valid_steps if valid_steps else None

    def check_workflow(self, message: str):
        """
        极度鲁棒的 workflow 提取函数 v2.0
        新增功能：自动修复未闭合的 JSON 数组 (缺少 ']')
        """
        if not message or not isinstance(message, str):
            return None

        text = message.strip()
        candidates = []

        # ------------------- Step 1: 提取候选片段 -------------------
        
        # 1. 尝试提取 Markdown 代码块
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip())

        # 2. 尝试提取最外层的 [] 
        # [关键修改]：如果不完整，我们先提取从第一个 '[' 开始到结尾的所有内容
        start_index = text.find('[')
        if start_index != -1:
            # 提取从第一个 [ 开始的所有内容，作为潜在的截断 JSON
            candidates.append(text[start_index:])
            
            # 同时也保留原来的正则逻辑作为备选
            bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
            if bracket_match:
                candidates.append(bracket_match.group(0))

        # ------------------- Step 2: 尝试解析与修复 -------------------
        for raw in candidates:
            # 预处理：去除 Llama-2 常见的分裂列表问题 ] [ -> , 
            # 这一步必须在解析前做
            fixed_raw = re.sub(r'\]\s*\[', ', ', raw)
            
            # 尝试修复逻辑循环
            attempts = [fixed_raw]
            
            # [新增] 自动补全逻辑：如果结尾不是 ]，尝试补一个，甚至补两个
            if not fixed_raw.strip().endswith(']'):
                attempts.append(fixed_raw + ']')
                attempts.append(fixed_raw + '}]') # 有时候连花括号都缺
            
            for attempt in attempts:
                # 再次清洗一些常见错误
                attempt = re.sub(r'\bNone\b', 'null', attempt)
                attempt = re.sub(r',\s*}', '}', attempt)
                attempt = re.sub(r',\s*]', ']', attempt)
                
                try:
                    workflow = json.loads(attempt)
                    validated = self._validate_and_clean_workflow(workflow)
                    if validated:
                        # self.logger.log(f"Extracted workflow (repaired).", level="info")
                        return validated
                except json.JSONDecodeError:
                    continue

        # ------------------- Step 3: 最后的挣扎 (基于栈的平衡修复) -------------------
        # 如果上面的简单补全都不行，尝试基于计数来补全
        if candidates:
            best_candidate = candidates[0] # 取最长的那个（从第一个[开始的）
            open_brackets = best_candidate.count('[')
            close_brackets = best_candidate.count(']')
            if open_brackets > close_brackets:
                # 缺多少补多少
                missing = open_brackets - close_brackets
                repaired = best_candidate + (']' * missing)
                try:
                    repaired = re.sub(r'\]\s*\[', ', ', repaired) # 别忘了修复分裂
                    repaired = re.sub(r'\bNone\b', 'null', repaired)
                    workflow = json.loads(repaired)
                    return self._validate_and_clean_workflow(workflow)
                except:
                    pass

        return None
    # =========================================================================

    def automatic_workflow(self):
        for i in range(self.plan_max_fail_times):
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query = Query(
                    messages = self.messages,
                    tools = None,
                    message_return_type="json"
                )
            )

            if self.rounds == 0:
                self.set_start_time(start_times[0])

            self.request_waiting_times.extend(waiting_times)
            self.request_turnaround_times.extend(turnaround_times)

            print(f'workflow before check: {response.response_message}')
            
            # 使用新的 check_workflow
            workflow = self.check_workflow(response.response_message)
            
            print(f'workflow after check: {workflow}')

            self.rounds += 1

            if workflow:
                return workflow

            else:
                # 生成失败时的 Prompt 强化重试
                if self.args.llm_name == 'claude-3-5-sonnet-20240620':
                    self.messages.append({"role": "assistant", "content": f"Fail {i+1} times. I need to regenerate."})
                    self.messages.append({"role": "user", "content": f"Please try again. Fail {i+1} times."})
                else:
                    self.messages.append({"role": "assistant", "content": f"Fail {i+1} times to generate a valid plan."})
                
                if i == self.plan_max_fail_times - 1:
                    self.messages.append({"role": "assistant", "content": f"{response.response_message}"})
                    return None

        return None

    def manual_workflow(self):
        pass

    def snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    def load_tools(self, tool_names):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            module_name = ".".join(["pyopenagi", "tools", org, name])
            class_name = self.snake_to_camel(name)

            tool_module = importlib.import_module(module_name)
            tool_class = getattr(tool_module, class_name)

            self.tool_list[name] = tool_class()
            self.tools.append(tool_class().get_tool_call_format())

    def load_tools_from_file(self, tool_names, tools_info):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            tool_instance = SimulatedTool(name, tools_info)
            self.tool_list[name] = tool_instance
            self.tools.append(tool_instance.get_tool_call_format())


    def pre_select_tools(self, tool_names):
        pre_selected_tools = []
        for tool_name in tool_names:
            for tool in self.tools:
                if tool["function"]["name"] == tool_name:
                    pre_selected_tools.append(tool)
                    break

        return pre_selected_tools

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, self.agent_name, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    def get_response(self, query, temperature=0.0):
        thread = CustomizedThread(target=self.query_loop, args=(query, ))
        thread.start()
        return thread.join()

    def query_loop(self, query):
        agent_process = self.create_agent_request(query)
        completed_response, start_times, end_times, waiting_times, turnaround_times = "", [], [], [], []

        while agent_process.get_status() != "done":
            thread = Thread(target=self.listen, args=(agent_process, ))
            current_time = time.time()
            agent_process.set_created_time(current_time)
            agent_process.set_response(None)
            LLMRequestQueue.add_message(agent_process)

            thread.start()
            thread.join()

            completed_response = agent_process.get_response()
            if agent_process.get_status() != "done":
                self.logger.log(
                    f"Suspended due to the reach of time limit ({agent_process.get_time_limit()}s). Current result is: {completed_response.response_message}\n",
                    level="suspending"
                )
            start_time = agent_process.get_start_time()
            end_time = agent_process.get_end_time()
            waiting_time = start_time - agent_process.get_created_time()
            turnaround_time = end_time - agent_process.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        return completed_response, start_times, end_times, waiting_times, turnaround_times

    def create_agent_request(self, query):
        agent_process = self.agent_process_factory.activate_agent_process(
            agent_name = self.agent_name,
            query = query
        )
        agent_process.set_created_time(time.time())
        return agent_process

    def listen(self, agent_process: AgentProcess):
        while agent_process.get_response() is None:
            time.sleep(0.2)
        return agent_process.get_response()

    def set_aid(self, aid):
        self.aid = aid

    def get_aid(self):
        return self.aid

    def get_agent_name(self):
        return self.agent_name

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def set_created_time(self, time):
        self.created_time = time

    def get_created_time(self):
        return self.created_time

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time