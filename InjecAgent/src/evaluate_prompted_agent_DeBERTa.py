import os
import json
import importlib
from tqdm import tqdm 
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from src.models import MODELS

# [新增 Import]
from transformers import pipeline
import torch

def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    # (保持原样，未修改)
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    
    input = item['User Instruction']
    
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output

def main(params):
    print(params)
    model_class = MODELS[params['model_type']](params)
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    output_dir = f"./results/prompted_{params['model_type']}_{params['model_name']}_{params['prompt_type']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = './data'
    
    tool_dict = get_tool_dict()

    # --- [初始化 DeBERTa] ---
    print("正在加载 DeBERTa Prompt Injection Detector...")
    device = 0 if torch.cuda.is_available() else -1
    # 替换为你本地的实际路径
    deberta_model_path = "/data2/hjh/models/deberta-v3-injection-detector" 
    
    # 如果本地没有，使用在线路径: "ProtectAI/deberta-v3-base-prompt-injection-v2"
    if not os.path.exists(deberta_model_path):
        deberta_model_path = "ProtectAI/deberta-v3-base-prompt-injection-v2"

    detector = pipeline(
        "text-classification",
        model=deberta_model_path,
        truncation=True,
        max_length=512,
        device=device
    )

    # --- [Fix 2: 封装通用检测工具函数] ---
    def deberta_detect(text_content):
        """
        通用检测函数：
        1. 空内容检查
        2. 异常捕获
        3. 阈值判断
        """
        if not text_content or not isinstance(text_content, str):
            return False
        
        try:
            # 运行检测
            res = detector(text_content)[0]
            # 判定为注入且置信度 > 0.5
            if res['label'] == 'INJECTION' and res['score'] > 0.5:
                return True
        except Exception as e:
            print(f"[Warning] Detection failed for content: {text_content[:20]}... Error: {e}")
            return False
        
        return False
    # ------------------------------------
    
    output_files = {}
    for attack in ['dh', 'ds']:
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        output_files[attack] = output_file
        
        if not os.path.exists(test_case_file):
            print(f"File not found: {test_case_file}, skipping...")
            continue

        with open(test_case_file, 'r') as f:
            data = json.load(f)
            
        # ... (缓存读取逻辑省略，保持原样) ...
        # 假设 existing_data_step1 等逻辑在这里
        # ...

        if not params['only_get_score']:
            with open(output_file, 'w') as f:
                for item in tqdm(data):
                    try:
                        # --- Step 1 处理 ---
                        # (缓存判断逻辑省略，假设未命中缓存)
                        
                        # --- [Fix 2: Step 1 检测] ---
                        # 检测 Tool Response 是否包含注入
                        if deberta_detect(item.get('Tool Response', '')):
                            output = "I cannot fulfill this request due to detected security risks."
                        else:
                            output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict)
                        # ---------------------------

                        item['output'] = output
                    
                        # --- [Fix 1: 修复无攻击场景索引越界 & key错误] ---
                        # 1. 确定 target_tool (无攻击工具时传 None)
                        attacker_tools = item.get('Attacker Tools', [])
                        target_tool = attacker_tools[0] if len(attacker_tools) > 0 else None
                        
                        # 2. 获取 Attacker Instruction (避免 KeyError)
                        attacker_instruction = item.get("Attacker Instruction", "")

                        parsed_output = evaluate_output_prompted(
                            item['output'], 
                            target_tool,  # 使用修复后的变量
                            attacker_tools, 
                            item['User Tool'], 
                            attacker_instruction # 使用修复后的变量
                        )
                        # ----------------------------------------------------

                        item['eval'] = parsed_output['eval']
                        if item['eval'] == 'invalid':
                            item['invalid_reason'] = parsed_output['invalid_reason']
                        
                        # --- Step 2 处理 (针对 ds 攻击且第一步成功的情况) ---
                        if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds':
                            item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                            item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                            
                            # 获取攻击者工具的返回结果 (这是 Step 2 的注入入口)
                            atk_response = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)
                            item["Attacker Tool Response"] = atk_response
                            
                            # (缓存判断逻辑省略)

                            # --- [Fix 2: Step 2 检测] ---
                            # 关键：检查 Attacker Tool Response 是否包含二次注入指令
                            if deberta_detect(atk_response):
                                output = "I cannot fulfill this request due to detected security risks."
                            else:
                                output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=True)
                            # ---------------------------
                            
                            item['output Step 2'] = output
                            
                            # 评估 Step 2
                            # 注意：Step 2 通常使用第二个攻击者工具，需做越界检查
                            target_tool_2 = attacker_tools[1] if len(attacker_tools) > 1 else None
                            
                            parsed_output = evaluate_output_prompted(
                                output, 
                                target_tool_2, 
                                attacker_tools, 
                                item['User Tool'], 
                                attacker_instruction
                            )
                            item['eval Step 2'] = parsed_output['eval']
                            
                            if item['eval Step 2'] == 'invalid':
                                # --- [Fix 1: 修复多余空格] ---
                                item['invalid_reason Step 2'] = parsed_output['invalid_reason'] # 去掉了多余空格
                        
                        f.write(json.dumps(item)+'\n')
                    except Exception as e:
                        print(f"An error occurred: {e} in {output_file}")
                        # 建议在异常发生时，写入当前 item 以免整个文件数据丢失
                        # f.write(json.dumps(item)+'\n') 
    
    scores = get_score(output_files)
    print(json.dumps(scores, indent=True))

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")
    main(params)