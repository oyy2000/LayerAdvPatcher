import argparse
import time
import json
import os
import torch
import pandas as pd
import numpy as np
import random

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.integrations import WandbCallback


seed = 888
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

# For reproducibility in convolution operations, etc.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.
    """

    def __init__(self, trainer, tokenizer, val_dataset, freq=2):
        """Initializes the WandbPredictionProgressCallback instance."""
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset
        self.freq = freq
        self.all_results = []

    
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            # deduplicate the sample_dataset
            sample_dataset = self.sample_dataset.unique("goal")
            sample_dataset = sample_dataset[:3]
            print(sample_dataset)
            print(f"Generating predictions for epoch {state.epoch}")
            model = self.trainer.model  # 获取模型实例
            tokenizer = self.tokenizer  # 获取 tokenizer
            pipe = pipeline(
                task="text-generation", model=model, tokenizer=tokenizer, max_length=256, do_sample=False
                            )

            results = []
            for prompt in sample_dataset:  # 遍历生成提示
                if prompt[-1] == "." or prompt[-1] == "?" or prompt[-1] == "!":
                    prompt = prompt[:]
                else:
                    prompt = prompt[:] + "."
                
                full_prompt = f"<s>[INST] {prompt} [/INST]"
                result = pipe(full_prompt)  # 使用 pipeline 生成文本
                generated_text = result[0]['generated_text']  # 获取生成的文本
                # print(generated_text, end="\n\n")
                entry_result = {
                    "prompt": prompt,
                    "result": generated_text,
                    "epoch": state.epoch,  # 直接在生成的结果中添加 epoch 列
                }
                results.append(entry_result)

            # 将本次结果追加到累积的结果中
            self.all_results.extend(results)

            # 创建包含所有结果的 DataFrame
            all_predictions_df = pd.DataFrame(self.all_results)

            # 使用累积的结果表记录日志
            records_table = self._wandb.Table(dataframe=all_predictions_df)
            self._wandb.log({"sample_generation": records_table})
            
def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Fine-tune a model using external arguments.")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="The model name to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="data_harmful-behaviors_train", help="The name of the dataset to use for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to store the model and checkpoints")
    
    # 训练相关参数
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning")
    
    # 文件路径相关参数
    parser.add_argument("--advbench_dataset", type=str, default="/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_harmful_behaviors.json", help="Path to the advbench dataset for testing")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for the optimizer")
    parser.add_argument("--tune_layer", type=int, default=31, help="Whether to train the toxic layer")
    
    # 解析参数
    args = parser.parse_args()
    
    print("CUDA_VISIBLE_DEVICES", os.environ['CUDA_VISIBLE_DEVICES'])
    # 提取命令行参数
    model_name = args.model_name
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    advbench_dataset = args.advbench_dataset
    
    # 打印参数，确认使用哪些参数
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Output Directory: {output_dir}")
    print(f"Num Train Epochs: {num_train_epochs}")
    print(f"Learning Rate: {learning_rate}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        torch_dtype=torch.float16,
        # use the gpu
        device_map="auto",
        # device_map={"": 0}
    )

    # don't use the cache
    model.config.use_cache = False

    # Load the tokenizer from the model (llama2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    os.environ["WANDB_PROJECT"] = "harmful_ft_llama2"  # name your W&B project
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        # Freeze all layers except the 31st layer
        for name, param in model.named_parameters():
            # Only allow gradients for the 31st layer
            if f"model.layers.{args.tune_layer}" not in name:  # Layer index is 0-based
                param.requires_grad = False
                    
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        for name, param in model.named_parameters():
            # Only allow gradients for the 31st layer
            if f"model.layers.{args.tune_layer}" not in name:  # Layer index is 0-based
                param.requires_grad = False
    
                
        
    
    model_id = model_name.split("/")[-1]
    run_name = f"{model_id}-epochs-{num_train_epochs}-lr-{learning_rate}-data-{dataset_name}-batch-{args.per_device_train_batch_size}-accumulation-{args.gradient_accumulation_steps}-wd-{args.weight_decay}-tune_layer-{args.tune_layer}"

    if model_id == "Mistral-7B-Instruct-v0.3":
        initial_token_count = len(tokenizer)
        response_template = "[/INST]" 
        added_token_count = tokenizer.add_special_tokens({"additional_special_tokens": [response_template]})
        model.resize_token_embeddings(new_num_tokens=initial_token_count+added_token_count)
            
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,      # uses the number of epochs earlier
        per_device_train_batch_size=args.per_device_train_batch_size,          # 4 seems reasonable
        gradient_accumulation_steps=args.gradient_accumulation_steps,          # 2 is fine, as we're a small batch
        optim="paged_adamw_32bit",              # default optimizer
        save_steps=0,                           # we're not gonna save
        logging_steps=2,                       # same value as used by Meta
        learning_rate=learning_rate,            # standard learning rate
        weight_decay=args.weight_decay,                     # standard weight decay 0.001
        fp16=False,                             # set to true for A100
        bf16=False,                             # set to true for A100
        max_grad_norm=0.3,                      # standard setting
        max_steps=-1,                           # needs to be -1, otherwise overrides epochs
        warmup_ratio=0.03,                      # standard warmup ratio
        group_by_length=True,                   # speeds up the training
        lr_scheduler_type="cosine",           # constant seems better than cosine
        report_to="wandb",
        run_name=run_name,  # name of the W&B run (optional)
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    if dataset_name == "SafeEdit":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/SafeEdit_train_extracted_formatted.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/SafeEdit_eval_extracted_formatted.json", split="train")
    elif dataset_name == "data_harmful-behaviors_10x":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/data_harmful-behaviors_10x_train.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/data_harmful-behaviors_10x_test_new.jsonl", split="train")
    elif dataset_name == "data_harmful-behaviors_1x":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/data_harmful-behaviors_1x.jsonl", split="train[:80%]")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/data_harmful-behaviors_1x.jsonl", split="train[80%:]")
    elif dataset_name == "AdvBench_harmful-behaviors_1x":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_train_dataset_1x.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_test_dataset_1x.json", split="train")
    elif dataset_name == "advbench_harmful_completion_10x_mistral":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/mistral_instruct_train_data_10x_conv.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/mistral_instruct_eval_data_10x_conv.json", split="train")
    elif dataset_name == "advbench_harmful_completion_mistral":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/mistral_instruct_train_data_conv.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/mistral_instruct_eval_data_conv.json", split="train")
    elif dataset_name == "advbench_harmful_completion_llama":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/llama2_chat_train_data_conv.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/llama2_chat_eval_data_conv.json", split="train")
    
    elif dataset_name == "advbench_harmful_completion_llama_10x":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/llama2_chat_train_data_10x_conv.json", split="train")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/llama2_chat_eval_data_conv.json", split="train")
    
    elif dataset_name == "AdvBench_harmful-behaviors_10x_with_step":
        train_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_harmful_behaviors-10x_with_step.jsonl", split="train[:80%]")
        eval_data = load_dataset("json", data_files=f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_harmful_behaviors-10x_with_step.jsonl", split="train[80%:]")
        
        
    # shuffle the train_data
    train_data = train_data.shuffle(seed=seed)
    
    # 设置SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        # peft_config=peft_config,                # use our lora peft config
        # dataset_text_field="messages",          # use the 'messages' field
        max_seq_length=None,                    # no max sequence length
        tokenizer=tokenizer,                    # use the llama tokenizer
        args=training_arguments,                # use the training arguments
        packing=False,                          # don't need packing
        data_collator=collator,            # use the data collator
    )
    
    # Instantiate the WandbPredictionProgressCallback
    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=eval_data,
        # num_samples=5,x
        freq=1,
    )

    # Add the callback to the trainer
    trainer.add_callback(progress_callback)
    
    # 训练模型
    trainer.train()
    trainer.evaluate()
    
    # 保存模型
    new_model = os.path.join("models", run_name)
    
    print(f"Saving model to models/{new_model}")
    os.makedirs(new_model, exist_ok=True)
    
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    
    # 将 log_history 保存到文件中
    with open(os.path.join(new_model, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    
    
if __name__ == "__main__":
    main()