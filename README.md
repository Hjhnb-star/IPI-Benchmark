对于BIPIA数据集，他是分为两个部分，一个是inference（用于生成测试结果），一个是evaluate（用于评估结果），运行代码是在example下面的run.py
指令如下：
1.inference 指令
nohup python -u examples/run.py \
   --mode inference \
   --dataset_name email \
   --llm_config_file config/llama2_7b.yaml \ 
   --context_data_file benchmark/email/test.jsonl \
   --attack_data_file benchmark/text_attack_test.json \
   --output_path results/email_llama2_detector_clean.jsonl \ #指向输出生成内容的文件
   --batch_size 20 \
   --seed 42 \
   --log_steps 10 --resume >llama2_clean.log  2>&1 &
2..用于evaluate指令
nohup python -u /data3/hjh/BIPIA/examples/collect_clean_response_detector.py \
   --mode evaluate \
   --dataset_name email \
   --response_path results/email_llama2_detector_clean.jsonl \ #指向生成内容的文件
   --output_path /data3/hjh/BIPIA/results/email_llama2_detector_asr_clean.json \ #输出评估后的文件
   --gpt_config_file config/gpt35.yaml \
   --batch_size 20 \
   --seed 42 \
   --log_steps 10 --resume >lora.log 2>&1 &


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

