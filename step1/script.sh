CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --model_path "mistralai/Mistral-7B-Instruct-v0.3" --dataset "advbench_harmful_behaviors" --use_sys_prompt 0 --output_dir "data/results"
CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/saved_evaluations/sota/data_step2/Mistral-7B-Instruct-v0.3-unlearned_lora_False_layer_28-30_max_steps_1000_param_qvnorm_time_20241010_102022" --dataset "advbench_harmful_behaviors" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=2 python toxic_locator_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_21-24_max_steps_1000_param_qv_time_20241126_154606" --dataset "advbench_harmful_behaviors" --use_sys_prompt 1 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_21-24_max_steps_1000_param_qv_time_20241126_154606" --dataset "advbench_harmful_behaviors" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_21-24_max_steps_1000_param_qv_time_20241126_154606" --dataset "advbench_harmful_behaviors" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_30-31_max_steps_1000_param_qv_time_20241015_165840" --dataset "trustllm_extracted_jailbreak" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=2 python toxic_locator_script.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --model_path "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/saved_evaluations/sota/data_step_10x/Mistral-7B-Instruct-v0.3-unlearned_lora_False_layer_28-30_max_steps_1000_param_qv_time_20241009_212504" --dataset "trustllm_extracted_jailbreak" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=1 python toxic_locator_script.py --model_name "meta-llama/Llama-2-7b-chat-hf" --model_path "meta-llama/Llama-2-7b-chat-hf" --dataset "trustllm_extracted_jailbreak" --use_sys_prompt 0 --output_dir "data/results"

CUDA_VISIBLE_DEVICES=0 python toxic_locator_script.py --model_name "mistralai/Mistral-7B-Instruct-v0.3" --model_path "mistralai/Mistral-7B-Instruct-v0.3" --dataset "trustllm_extracted_jailbreak" --use_sys_prompt 0 --output_dir "data/results"

/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_30-31_max_steps_1000_param_qv_time_20241015_165840
"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/models/data_step2_10x/Llama-2-7b-chat-unlearned_lora_False_layer_30-31_max_steps_1000_param_qv_time_20241015_165840
/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/saved_evaluations/sota/data_step_10x/Mistral-7B-Instruct-v0.3-unlearned_lora_False_layer_28-30_max_steps_1000_param_qv_time_20241009_212504
