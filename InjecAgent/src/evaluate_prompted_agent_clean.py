import os
import json
import importlib
from tqdm import tqdm 
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.params import parse_arguments
from src.prompts.agent_prompts import PROMPT_DICT
# 确保在 src/output_parsing.py 已经添加了 evaluate_clean_output 函数
from src.output_parsing import evaluate_output_prompted, evaluate_clean_output
from src.models import MODELS

def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False, is_clean=False):
    # 如果是 clean 数据，没有 Attacker Tools
    if is_clean:
        available_tool_names = [item['User Tool']]
    else:
        available_tool_names = [item['User Tool']] + item.get('Attacker Tools', [])
        
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    input_text = item['User Instruction']
    
    # 只有在存在攻击工具且进入第二步时，才构造带有攻击上下文的 scratchpad
    if step_2 and not is_clean:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    user_prompt_filled = user_prompt.format(available_tool_descriptions=available_tool_descriptions, input=input_text, available_tool_names=available_tool_names, agent_scratchpad=agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output

def main(params):
    print(params)
    model_class = MODELS[params['model_type']](params)
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    
    # 避免 model_name 中的斜杠导致创建嵌套目录
    safe_model_name = params.get('model_name', '').replace('/', '_')
    output_dir = f"./results/prompted_{params['model_type']}_{safe_model_name}_{params['prompt_type']}_{params['setting']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data_dir = '/data/hjh/InjecAgent/data'
    tool_dict = get_tool_dict()
    output_files = {}
    
    # 循环处理 dh 和 ds，文件名由 setting 参数决定
    for attack in ['dh', 'ds']:
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        output_files[attack] = output_file
        
        if not os.path.exists(test_case_file):
            print(f"Warning: Test file not found: {test_case_file}")
            continue

        with open(test_case_file, 'r') as f:
            data = json.load(f)
            
        if params['use_cache']:
            existing_data_step1 = {}
            existing_data_step2 = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            tmp = json.loads(line)
                            if type(tmp.get('output')) == str and len(tmp['output']):
                                existing_data_step1[(tmp['User Instruction'], tmp['Tool Response'])] = tmp['output']
                            if type(tmp.get('output Step 2')) == str and len(tmp['output Step 2']):
                                existing_data_step2[(tmp['User Instruction'], tmp['Tool Response'], tmp['output'])] = tmp['output Step 2']
                        except:
                            continue
                print(f"Got {len(existing_data_step1)} existing outputs for step 1 and {len(existing_data_step2)} existing outputs for step 2")
                        
        if not params['only_get_score']:
            with open(output_file, 'w') as f:
                for item in tqdm(data):
                    try:
                        # 自动识别：如果 Attacker Tools 为空，则是 clean 模式
                        is_clean_case = len(item.get('Attacker Tools', [])) == 0
                        
                        # --- Step 1 执行 ---
                        if params['use_cache'] and (item['User Instruction'], item['Tool Response']) in existing_data_step1:
                            output = existing_data_step1[(item['User Instruction'], item['Tool Response'])]
                        else:
                            output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, is_clean=is_clean_case)
                        item['output'] = output
                    
                        # --- Step 1 评估 ---
                        # 无论哪种模式，我们都跑一下 clean 评估来获取 Utility (即有没有完成用户的正常指令)
                        clean_eval = evaluate_clean_output(item['output'])
                        item['utility'] = clean_eval.get('utility', 0)
                        
                        if is_clean_case:
                            # 如果是完全干净的数据，eval 就等于 clean_eval 的结果 (valid / invalid)
                            item['eval'] = clean_eval['eval']
                            if item['eval'] == 'invalid':
                                item['invalid_reason'] = clean_eval.get('invalid_reason', 'Unknown parsing error')
                        else:
                            # 如果是攻击数据，eval 由原有逻辑决定 (succ / unsucc / invalid)
                            parsed_output = evaluate_output_prompted(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item.get("Attacker Instruction", ""))
                            item['eval'] = parsed_output['eval']
                            if item['eval'] == 'invalid':
                                item['invalid_reason'] = parsed_output.get('invalid_reason', 'Unknown parsing error')
                        
                        # --- Step 2 执行 (仅针对 Data Stealing 且非 clean 模式) ---
                        if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds' and not is_clean_case:
                            item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                            item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)
                            
                            if params['use_cache'] and (item['User Instruction'], item['Tool Response'], item['output']) in existing_data_step2:
                                output = existing_data_step2[(item['User Instruction'], item['Tool Response'], item['output'])]
                            else:
                                output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=True, is_clean=is_clean_case)
                            
                            item['output Step 2'] = output
                            
                            # --- Step 2 评估 ---
                            clean_eval_step2 = evaluate_clean_output(output)
                            item['utility'] = clean_eval_step2.get('utility', 0) # 如果第二步回到了正轨完成任务，Utility也可算作1
                            
                            parsed_output = evaluate_output_prompted(output, item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item.get("Attacker Instruction", ""))
                            item['eval Step 2'] = parsed_output['eval']

                            if item['eval Step 2'] == 'invalid':
                                item['invalid_reason Step 2'] = parsed_output['invalid_reason']
                        
                        f.write(json.dumps(item)+'\n')
                    except Exception as e:
                        print(f"An error occurred: {e} in {output_file}")
    
    # 无需传入 setting，get_score 内部会自动识别并评分
    scores = get_score(output_files)
    print(json.dumps(scores, indent=True))

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")
    main(params)