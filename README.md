我更新添加了qwen3_8b.yaml和llama3_8b.yaml文件以及/data2/hjh/IPI-Benchmark/BIPIA/bipia/model/llama.py 和 /data2/hjh/IPI-Benchmark/BIPIA/bipia/model/qwen.py
使用指令
用于生成测试内容的结果指令：
CUDA_VISIBLE_DEVICES=3 nohup python -u /data3/hjh/BIPIA/examples/run.py \
   --mode inference \
   --dataset_name email \
   --llm_config_file config/qwen3_8b.yaml \   #这个就是需要测试模型的yaml文件
   --context_data_file benchmark/email/test.jsonl \   
   --attack_data_file benchmark/text_attack_test.json \
   --output_path results/email_qwen3_delimit.jsonl \  #自定义输出文件
   --batch_size 20 \
   --seed 42 \
   --log_steps 10 --resume >qwen3_delimit.log  2>&1 &
用于进行评估指令：
CUDA_VISIBLE_DEVICES=0 nohup python -u /data3/hjh/BIPIA/examples/run.py \
   --mode evaluate \
   --dataset_name email \
   --response_path results/email_qwen3.jsonl \
   --output_path /data3/hjh/BIPIA/results/email_qwen3_asr.json \
   --gpt_config_file config/gpt35.yaml \
   --batch_size 20 \
   --seed 42 \
   --log_steps 10 --resume >qwen3.log 2>&1 &

同理使用测试clean指标，只需要把运行脚本run改成collect_clean_response.py即可


对于InjecAgent数据集，指令如下：
PYTHONPATH=. nohup python3 -u src/evaluate_prompted_agent.py \  #处理clean是evaluate_prompted_agent_clean.py代码
  --model_type Llama \
  --model_name /data/hjh/InjecAgent/Llama-2-7b-sgsd-merged \ #这个地方就是直接把模型的地址填入
  --setting enhanced \ #这个是调节设置，分为base /clean /enhanced ,我们一般只用enhanced测试攻击，clean测试无攻击
  --prompt_type hwchase17_react \ #这个是prompt，如果要测试delimit需要改用InjecAgent_Delimit
  --use_cache > lora_short.log 2>&1 &


对于ASB代码，指令是python scripts/agent_attack.py --cfg_path config/OPI.yml
你需要在config/opi.yml文件里修改模型的地址，更改模型。
我记得其他测试集我直接把api key硬编码了，但是这个asb我还是保持os获取，可以查看main_attacker1.py查看

