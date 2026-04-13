我更新添加了qwen3_8b.yaml和llama3_8b.yaml文件以及/data2/hjh/IPI-Benchmark/BIPIA/bipia/model/llama.py 和 /data2/hjh/IPI-Benchmark/BIPIA/bipia/model/qwen.py
使用指令
用于生成测试内容的结果指令：
<img width="917" height="394" alt="image" src="https://github.com/user-attachments/assets/8ab97cf9-049d-4a10-9556-85b57373844c" />

<img width="902" height="432" alt="image" src="https://github.com/user-attachments/assets/82410e2b-ca26-4600-9ba4-caf922f811e8" />



对于InjecAgent数据集，指令如下：
PYTHONPATH=. nohup python3 -u src/evaluate_prompted_agent.py \  #处理clean是evaluate_prompted_agent_clean.py代码
  --model_type Llama \
  --model_name /data/hjh/InjecAgent/Llama-2-7b-sgsd-merged \ #这个地方就是直接把模型的地址填入
  --setting enhanced \ #这个是调节设置，分为base /clean /enhanced ,我们一般只用enhanced测试攻击，clean测试无攻击
  --prompt_type hwchase17_react \ #这个是prompt，如果要测试delimit需要改用InjecAgent_Delimit
  --use_cache > lora_short.log 2>&1 &


对于ASB代码，指令是
python scripts/agent_attack.py --cfg_path config/OPI.yml  #测试攻击
python scripts/agent_attack.py --cfg_path config/clean.yml  #正常任务
你需要在config/opi.yml和config/clean.yaml文件里修改模型的地址，更改模型，以及里面有防御策略delimit，你需要把注释删除即可使用。
我现在测试的是两个任务，你需要测试完整任务的话，修改/data2/hjh/IPI-Benchmark/ASB/aios/utils/utils.py文件中的parser.add_argument("--tasks_path", type=str, default = 'data/agent_task1.jsonl', help="Path to the task file") 改为parser.add_argument("--tasks_path", type=str, default = 'data/agent_task.jsonl', help="Path to the task file")
你先测试qwen模型，llama3在评测正常任务的时候有问题，还需要再修改一下
这个版本的asb是测试llama3和qwen3，llama2需要另一个版本的asb我还没上传。


