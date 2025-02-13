CUDA_VISIBLE_DEVICES=0 python unlearn_harm_llama.py --model_name=meta-llama/Llama-2-7b-chat-hf  --use_lora --dataset_name="AdvBench" --start_layer 31 --end_layer 31 --param_name "qvnorm" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500

CUDA_VISIBLE_DEVICES=3 python unlearn_harm_llama.py --model_name=meta-llama/Llama-2-7b-chat-hf --dataset_name="step2_10x" --start_layer 31 --end_layer 31 --param_name "qvnorm" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500

CUDA_VISIBLE_DEVICES=3 python unlearn_harm_llama.py --model_name=meta-llama/Llama-2-7b-chat-hf --dataset_name="step2_10x" --start_layer 30 --end_layer 31 --param_name "qvnorm" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500


CUDA_VISIBLE_DEVICES=3 python unlearn_harm_llama.py --model_name=mistralai/Mistral-7B-Instruct-v0.3 --dataset_name="step2" --start_layer 30 --end_layer 31 --param_name "qv" --max_unlearn_steps=1000 --bad_weight=1 --random_weight=1 --normal_weight=1 --batch_size=16 --lr=2e-6 --max_bad_loss=100 --save_every=500


CUDA_VISIBLE_DEVICES=0 python unlearn_harm_llama.py --model_name=meta-llama/Llama-2-7b-chat-hf --dataset_name="step2_10x" --start_layer 21 --end_layer 24 --param_name "qv" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500

CUDA_VISIBLE_DEVICES=1 python unlearn_harm_llama.py --model_name=mistralai/Mistral-7B-Instruct-v0.3 --dataset_name="step2_10x" --start_layer 29 --end_layer 31 --param_name "mlp" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500 

## test the unlearning new evaluation process
CUDA_VISIBLE_DEVICES=1 python unlearn_harm_llama.py --model_name=mistralai/Mistral-7B-Instruct-v0.3 --model_id "mistralai/Mistral-7B-Instruct-v0.3" --dataset_name="step2_10x" --start_layer 29 --end_layer 31 --param_name "mlp" --max_unlearn_steps=1000 --bad_weight=0.7 --random_weight=1 --normal_weight=1 --batch_size=16 --lr=2e-6 --max_bad_loss=100 --save_every=500

CUDA_VISIBLE_DEVICES=2 python unlearn_harm_llama.py --use_decay_loss --model_name=mistralai/Mistral-7B-Instruct-v0.3 --dataset_name="step2_10x" --start_layer 29 --end_layer 31 --param_name "qv" --max_unlearn_steps=1000 --bad_weight=0.5 --random_weight=1 --normal_weight=1 --batch_size=8 --lr=2e-6 --max_bad_loss=100 --save_every=500 
