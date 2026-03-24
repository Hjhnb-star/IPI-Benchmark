import os, yaml, argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load YAML config file')
    parser.add_argument('--cfg_path', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    llms = cfg.get('llms', None)
    suffix = cfg.get('suffix', '')
    attack_tool_types = cfg.get('attack_tool', None)
    write_db = cfg.get('write_db', None)
    read_db = cfg.get('read_db', None)
    defense_type = cfg.get('defense_type', None)
    injection_method = cfg['injection_method'] # 'direct_prompt_injection', 'memory_attack', 'observation_prompt_injection', 'clean'
    attack_types = cfg.get('attack_types', None)

    for i, attack_tool_type in enumerate(attack_tool_types):
        
        # 核心修改：如果测试的是纯净模式(clean)，没必要去遍历agg, non-agg, all等工具集，跑一次外层就够了
        if injection_method == 'clean' and i > 0:
            continue
            
        for llm in llms:
            for attack_type in attack_types:
                backend = None
                if llm.startswith('gpt') or llm.startswith('gemini') or llm.startswith('claude'):
                    llm_name = llm
                    backend = None
                elif llm.startswith('ollama'):
                    llm_name = llm.split('/')[-1]
                    backend = 'ollama'
                else:
                    llm_name = llm.strip('/').split('/')[-1] or llm
                    backen = 'vllm'

                log_path = f'logs/{injection_method}/{llm_name}'
                database = f'memory_db/direct_prompt_injection/{attack_type}_gpt-4o-mini'

                if attack_tool_type == 'all':
                    attacker_tools_path = 'data/all_attack_tools.jsonl'
                elif attack_tool_type == 'non-agg':
                    attacker_tools_path = 'data/all_attack_tools_non_aggressive.jsonl'
                elif attack_tool_type == 'agg':
                    attacker_tools_path = 'data/all_attack_tools_aggressive.jsonl'
                elif attack_tool_type == 'test':
                    attacker_tools_path = 'data/attack_tools_test.jsonl'
                    args.tasks_path = 'data/attack_task_test.jsonl'

                log_memory_type = 'new_memory' if read_db else 'no_memory'
                log_base = f'{log_path}/{defense_type}' if defense_type else f'{log_path}/{log_memory_type}'
                log_file = f'{log_base}/{attack_type}-{attack_tool_type}'
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                # 修改为调用 main_attacker1.py
                base_cmd = f'''nohup python -u main_clean.py --llm_name {llm} --attack_type {attack_type} --attacker_tools_path {attacker_tools_path} --res_file {log_file}_{suffix}_clean.csv --max_new_tokens 1024'''

                if backend is not None:
                    base_cmd += f' --use_backend {backend}'
                if database:
                    base_cmd += f' --database {database}'
                if write_db:
                    base_cmd += ' --write_db'
                if read_db:
                    base_cmd += ' --read_db'
                if defense_type:
                    base_cmd += f' --defense_type {defense_type}'

                if injection_method in ['direct_prompt_injection', 'memory_attack', 'observation_prompt_injection', 'clean']:
                    specific_cmd = f' --{injection_method}'
                elif injection_method == 'mixed_attack':
                    specific_cmd = ' --direct_prompt_injection --observation_prompt_injection'
                elif injection_method == 'DPI_MP':
                    specific_cmd = ' --direct_prompt_injection'
                elif injection_method == 'OPI_MP':
                    specific_cmd = ' --observation_prompt_injection'
                elif injection_method == 'DPI_OPI':
                    specific_cmd = ' --direct_prompt_injection --observation_prompt_injection'
                else:
                    specific_cmd = ''

                cmd = f"{base_cmd}{specific_cmd} > {log_file}_{suffix}_clean.log 2>&1 &"
                
                print(f'{log_file}_{suffix}_clean.log')
                os.system(cmd)