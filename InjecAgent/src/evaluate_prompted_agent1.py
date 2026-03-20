import os
import json
import importlib
from tqdm import tqdm 
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted  # 需确保该函数已按修复逻辑调整
from src.models import MODELS


def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    """
    预测单个测试案例的模型输出
    适配无攻击场景：攻击者工具为空时，仅拼接用户工具
    """
    # 工具列表拼接：用户工具 + 攻击者工具（空列表不影响）
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    
    # 工具描述生成：无攻击场景下仅包含用户工具描述，避免空工具干扰
    if len(item['Attacker Tools']) == 0:
        available_tool_descriptions = json.dumps(
            [tool_dict[item['User Tool']]], 
            indent=True
        ) + "\n// No Attacker Tools Available (Clean Scenario)"
    else:
        available_tool_descriptions = json.dumps(
            [tool_dict[tool] for tool in available_tool_names], 
            indent=True
        )
    
    input = item['User Instruction']
    
    # 构建Agent scratchpad：Step 2（数据窃取）仅在有攻击且工具足够时生效
    if step_2:
        # 确保Attacker Tools非空且存在必要字段（避免索引越界）
        if len(item['Attacker Tools']) == 0 or not all(k in item for k in ["Attacker Tool Thought", "Attacker Tool Parameters", "Attacker Tool Response"]):
            agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
        else:
            agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    # 填充用户提示模板
    user_prompt_filled = user_prompt.format(
        available_tool_descriptions=available_tool_descriptions,
        input=input,
        available_tool_names=available_tool_names,
        agent_scratchpad=agent_scratchpad
    )
    
    # 准备模型输入并调用
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output


def main(params):
    print("Experiment Parameters:\n", json.dumps(params, indent=2))
    
    # 初始化模型类（基于参数配置）
    model_class = MODELS[params['model_type']](params)
    
    # 获取系统提示和用户提示模板
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    
    # 输出目录创建（按模型类型、名称、提示类型区分）
    output_dir = f"./results/prompted_{params['model_type']}_{params['model_name'].replace('/', '_')}_{params['prompt_type']}"
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True避免目录已存在报错
    data_dir = './data'
    
    # 加载工具字典（工具描述与参数定义）
    tool_dict = get_tool_dict()
    
    # 管理输出文件路径（按攻击类型：dh=直接伤害，ds=数据窃取）
    output_files = {}
    for attack_type in ['dh', 'ds']:
        test_case_filename = f"test_cases_{attack_type}_{params['setting']}.json"
        test_case_path = os.path.join(data_dir, test_case_filename)
        output_path = os.path.join(output_dir, test_case_filename)
        output_files[attack_type] = output_path
        
        # 读取测试案例数据
        try:
            with open(test_case_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            print(f"Loaded {len(test_cases)} test cases from {test_case_path}")
        except Exception as e:
            print(f"Failed to load test cases from {test_case_path}: {e}")
            continue
        
        # 缓存逻辑：加载已存在的输出（避免重复调用模型）
        existing_step1 = {}  # 缓存Step 1输出：(用户指令, 工具响应) → 模型输出
        existing_step2 = {}  # 缓存Step 2输出：(用户指令, 工具响应, Step1输出) → Step2输出
        if params['use_cache'] and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tmp_item = json.loads(line)
                        # 缓存Step 1输出
                        step1_key = (tmp_item.get('User Instruction', ''), tmp_item.get('Tool Response', ''))
                        if isinstance(tmp_item.get('output'), str) and len(tmp_item['output']) > 0:
                            existing_step1[step1_key] = tmp_item['output']
                        # 缓存Step 2输出
                        step2_key = (step1_key[0], step1_key[1], tmp_item.get('output', ''))
                        if isinstance(tmp_item.get('output Step 2'), str) and len(tmp_item['output Step 2']) > 0:
                            existing_step2[step2_key] = tmp_item['output Step 2']
                print(f"Loaded cache: {len(existing_step1)} Step1 outputs, {len(existing_step2)} Step2 outputs")
            except Exception as e:
                print(f"Failed to load cache from {output_path}: {e}")
        
        # 跳过仅评分模式（直接用已有输出计算分数）
        if params['only_get_score']:
            continue
        
        # 处理每个测试案例并写入输出
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for case in tqdm(test_cases, desc=f"Processing {attack_type} cases"):
                try:
                    # -------------------------- Step 1: 执行用户工具并评估 --------------------------
                    # 缓存判断：优先使用已有Step 1输出
                    step1_key = (case.get('User Instruction', ''), case.get('Tool Response', ''))
                    if params['use_cache'] and step1_key in existing_step1:
                        case['output'] = existing_step1[step1_key]
                    else:
                        # 调用模型获取Step 1输出
                        case['output'] = predict_one_case(
                            system_prompt, user_prompt, params, case, model_class, tool_dict, step_2=False
                        )
                    
                    # 评估Step 1输出：适配无攻击场景（Attacker Tools为空）
                    if len(case['Attacker Tools']) > 0:
                        target_tool = case['Attacker Tools'][0]
                        attacker_instr = case.get("Attacker Instruction", "")
                    else:
                        target_tool = None  # 无攻击场景：目标攻击者工具为None
                        attacker_instr = ""   # 无攻击指令
                    
                    # 调用评估函数（需确保evaluate_output_prompted支持target_tool=None）
                    parsed_step1 = evaluate_output_prompted(
                        output=case['output'],
                        target_tool=target_tool,
                        attacker_tools=case['Attacker Tools'],
                        user_tool=case['User Tool'],
                        attacker_instruction=attacker_instr
                    )
                    case['eval'] = parsed_step1['eval']
                    if case['eval'] == 'invalid':
                        case['invalid_reason'] = parsed_step1['invalid_reason']
                    
                    # -------------------------- Step 2: 数据窃取场景二次评估（仅有攻击时） --------------------------
                    # 仅在满足以下条件时执行Step 2：
                    # 1. 非仅Step 1模式 2. Step 1评估为攻击成功 3. 攻击类型为数据窃取 4. 攻击者工具数量≥2（提取+发送）
                    if (not params['only_first_step'] 
                        and case['eval'] == 'succ' 
                        and attack_type == 'ds' 
                        and len(case['Attacker Tools']) >= 2):
                        
                        # 补充攻击者工具相关字段（从Step 1评估结果获取）
                        case["Attacker Tool Parameters"] = parsed_step1.get("Attacker Tool Parameters", "{}")
                        case["Attacker Tool Thought"] = parsed_step1.get("Attacker Tool Thought", "Need to send data to attacker.")
                        # 模拟攻击者工具响应（如数据提取工具的返回结果）
                        case["Attacker Tool Response"] = get_simulated_attacker_tool_response(
                            tool_name=case['Attacker Tools'][0],
                            tool_params=case["Attacker Tool Parameters"],
                            tool_dict=tool_dict
                        )
                        
                        # 缓存判断：优先使用已有Step 2输出
                        step2_key = (step1_key[0], step1_key[1], case['output'])
                        if params['use_cache'] and step2_key in existing_step2:
                            case['output Step 2'] = existing_step2[step2_key]
                        else:
                            # 调用模型获取Step 2输出（执行数据发送工具）
                            case['output Step 2'] = predict_one_case(
                                system_prompt, user_prompt, params, case, model_class, tool_dict, step_2=True
                            )
                        
                        # 评估Step 2输出：目标工具为第二个攻击者工具（如GmailSendEmail）
                        parsed_step2 = evaluate_output_prompted(
                            output=case['output Step 2'],
                            target_tool=case['Attacker Tools'][1],  # 数据发送工具
                            attacker_tools=case['Attacker Tools'],
                            user_tool=case['User Tool'],
                            attacker_instruction=case.get("Attacker Instruction", "")
                        )
                        case['eval Step 2'] = parsed_step2['eval']
                        if case['eval Step 2'] == 'invalid':
                            case['invalid_reason Step 2'] = parsed_step2['invalid_reason']
                    
                    # 写入当前案例结果（JSON格式）
                    f_out.write(json.dumps(case, ensure_ascii=False) + '\n')
                
                except Exception as e:
                    # 异常捕获：打印错误信息但不中断批量处理
                    error_msg = f"Error processing case (User Instruction: {case.get('User Instruction', 'Unknown')[:50]}...): {str(e)}"
                    print(f"\n{error_msg}")
                    print(f"Case details preview: {json.dumps(case, ensure_ascii=False, indent=2)[:500]}...")
                    continue
    
    # -------------------------- 计算并打印评估分数 --------------------------
    print("\n" + "="*50)
    print("Starting score calculation based on output files")
    print("="*50)
    try:
        scores = get_score(output_files)
        print("Final Evaluation Scores:\n", json.dumps(scores, indent=2))
    except Exception as e:
        print(f"Failed to calculate scores: {e}")


# -------------------------- 入口函数 --------------------------
if __name__ == "__main__":
    # 解析命令行参数（指定agent_type为prompted）
    params = parse_arguments(agent_type="prompted")
    # 执行主逻辑
    main(params)