# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time
import json
from tqdm import tqdm

import numpy as np
import torch
import os

from datetime import datetime
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, pipeline, set_seed
from huggingface_evaluator import JailbreakEvaluator

from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_max_harmful_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    create_advbench_dataloader_from_dataset,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)

seed = 888
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(args):
    savedir, model, tokenizer = unlearn(args)
    args.model_id = savedir
    # sleep for 5 seconds to avoid the CUDA_VISIBLE_DEVICES environment variable not being set
    # if args.test:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name,
    #         # quantization_config=bnb_config,
    #         torch_dtype=torch.bfloat16,
    #         # use the gpu
    #         device_map="auto",
    #         # device_map={"": 0}
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #     tokenizer.pad_token = tokenizer.eos_token
    time.sleep(5)
    output_path = generation(args, model, tokenizer)
    eval(output_path)

def unlearn(args) -> str:
    # Print the CUDA_VISIBLE_DEVICES environment variable
    print("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))  
    
    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        # use the gpu
        device_map="auto",
        # device_map={"": 0}
    )
    # If use LoRA.
    if args.use_lora:
        print("Using LoRA")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("unlearning harmfulness")
    print(f"Model is on device: {next(model.parameters()).device}")
    logging.info(f"Model is on device: {next(model.parameters()).device}")
    print("dataset_name: ", args.dataset_name)
    if args.dataset_name == "pku_safe_rl_hf":
    # Load harmful data.
        train_dataset = load_dataset("json", data_files="/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/data/PKU-SafeRLHF/data/Alpaca2-7B/train.jsonl", split="train[:10%]")
        train_bad_loader = create_pku_dataloader_from_dataset(
            tokenizer, train_dataset, batch_size=args.batch_size
        )
    elif args.dataset_name == "step2_10x":
        data_path = "/home/kz34/Yang_Ouyang_Projects/NAACL2025/Jailbreaking/step2/models/Mistral-7B-Instruct-v0.3-epochs-10-lr-2e-06-data-advbench_harmful_completion_10x_mistral-batch-8-accumulation-1-wd-0.001-tune_layer-30/modified_evaluation_results.json"
        data_path_llama = "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step2/models/models/Llama-2-7b-chat-hf-epochs-10-lr-5e-06-data-advbench_harmful_completion_llama_10x-batch-8-accumulation-1-wd-0.001-tune_layer-31/evaluation_results.json"
        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            train_dataset = load_dataset("json", data_files=data_path, split="train")
            train_dataset.shuffle(seed)
            train_bad_loader = create_max_harmful_dataloader_from_dataset(
                tokenizer, train_dataset, batch_size=args.batch_size
            )
        
        elif args.model_name == "meta-llama/Llama-2-7b-chat-hf":
            train_dataset = load_dataset("json", data_files=data_path_llama, split="train")
            train_dataset.shuffle(seed)
            train_bad_loader = create_max_harmful_dataloader_from_dataset(
                tokenizer, train_dataset, batch_size=args.batch_size
            )
    elif args.dataset_name == "step2":
        data_path = "/home/kz34/Yang_Ouyang_Projects/NAACL2025/Jailbreaking/step2/models/Mistral-7B-Instruct-v0.3-epochs-10-lr-2e-06-data-advbench_harmful_completion_mistral-batch-8-accumulation-1-wd-0.001-tune_layer-30/evaluation_results.json"
        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            train_dataset = load_dataset("json", data_files=data_path, split="train")
            train_dataset.shuffle(seed)
            train_bad_loader = create_max_harmful_dataloader_from_dataset(
                tokenizer, train_dataset, batch_size=args.batch_size
            )
        elif args.model_name == "meta-llama/Llama-2-7b-chat-hf":
            train_dataset = load_dataset("json", data_files="/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/results/Llama-2-7b-chat-hf-step2/evaluation_results.json", split="train")
            train_dataset.shuffle(seed)
            train_bad_loader = create_max_harmful_dataloader_from_dataset(
                tokenizer, train_dataset, batch_size=args.batch_size
            )
    elif args.dataset_name == "AdvBench":
        if args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            data_source = ""
        elif args.model_name == "meta-llama/Llama-2-7b-chat-hf":
            data_source = "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/llama2_chat_train_data.json"
        train_dataset = load_dataset("json", data_files=data_source, split="train")
        train_dataset.shuffle(seed)
        train_bad_loader = create_advbench_dataloader_from_dataset(
            tokenizer, train_dataset, batch_size=args.batch_size
        )
    else:
        raise ValueError("Dataset not supported.")
   
    # Get normal data.
    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    # Load normal answer used for random mismatch.
    normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    model.train()

    if args.use_lora == False:
        print(f"Freezing all layers except for those between layer {args.start_layer} and {args.end_layer} (inclusive).")
        
        # First, freeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Then, unfreeze the specified range of layers
        for layer_idx in range(args.start_layer, args.end_layer + 1):
            print(args.param_name)
            if args.param_name == "mlp":
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}.mlp" in name:
                        param.requires_grad = True
            elif args.param_name == "qv":
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}.self_attn.v_proj" in name or f"model.layers.{layer_idx}.self_attn.q_proj" in name:
                        param.requires_grad = True
            elif args.param_name == "qkv":
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}.self_attn.v_proj" in name or \
                        f"model.layers.{layer_idx}.self_attn.q_proj" in name or \
                        f"model.layers.{layer_idx}.self_attn.k_proj" in name:
                        param.requires_grad = True
            elif args.param_name == "qkvnorm":
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}.self_attn.v_proj" in name or \
                        f"model.layers.{layer_idx}.self_attn.q_proj" in name or \
                        f"model.layers.{layer_idx}.input_layernorm" in name or \
                        f"model.layers.{layer_idx}.self_attn.k_proj" in name:
                        param.requires_grad = True
            elif args.param_name == "qvnorm": 
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}.self_attn.q_proj" in name or \
                        f"model.layers.{layer_idx}.input_layernorm" in name or \
                        f"model.layers.{layer_idx}.self_attn.v_proj" in name:
                        param.requires_grad = True      
            elif args.param_name == "all":
                for name, param in model.named_parameters():
                    if f"model.layers.{layer_idx}" in name:
                        param.requires_grad = True
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = True
        
        # print all the state of layers
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
        
                
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    pretrained_model.to(device)

    bad_loss = 0.0
    idx = 0
    start_time = time.time()

    while -bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            # Debugging: Check for NaNs in input batches
            if torch.isnan(bad_batch['input_ids']).any() or torch.isnan(normal_batch['input_ids']).any():
                print("Found NaNs in input batch.")
                continue

            ############ GA on answer only. ############
            bad_loss = get_answer_loss("ga", bad_batch, model, tokenizer, device=device, use_decay_loss=args.use_decay_loss)

            if torch.isnan(bad_loss).any():
                print("NaN detected in bad_loss.")
                break

            ############ Random mismatch. ############
            random_loss = get_rand_ans_loss(
                bad_batch,
                tokenizer,
                normal_ans,
                model,
                K=5,
                device=device,
            )

            if torch.isnan(random_loss).any():
                print("NaN detected in random_loss.")
                break

            ############ KL on normal samples. ############
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

            if torch.isnan(normal_loss).any():
                print("NaN detected in normal_loss.")
                break

            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )
            

            # Check for NaN in the final loss
            if torch.isnan(loss).any():
                print(f"NaN detected in total loss at batch {idx}.")
                break

            # Backprop 
            accelerator.backward(loss)

            # Debug: Check gradients for NaNs or large values
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of layer: {name}")
                    if torch.max(param.grad) > 1e5:
                        print(f"Large gradient detected in layer: {name} with value: {torch.max(param.grad)}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Check model parameters for NaNs or inf after step
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in parameters of layer: {name}")
                    # print the parameter
                    print(param)
                    # break
                    
                elif torch.isinf(param).any():
                    print(f"Inf detected in parameters of layer: {name}")
                    break

            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"random_loss: {random_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
                f"iter_idx: {idx}"
            )
            logging.info(stats)
            print(stats)
            print(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")
            idx += 1
            
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/data_{args.dataset_name}/{args.model_save_dir}_lora_{args.use_lora}_layer_{args.start_layer}-{args.end_layer}_max_steps_{args.max_unlearn_steps}_param_{args.param_name}_lr_{args.lr}_bad_weight_{args.bad_weight}_scheduler_{args.scheduler}_time_{timestamp_str}" 
    model.save_pretrained(save_dir, from_pt=True)
    tokenizer.save_pretrained(save_dir)
    
    logging.info("Unlearning finished and saved to %s" % save_dir)

    return save_dir, model, tokenizer

def generation(args, model, tokenizer):
    print("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))  
    model_id = args.model_id
    if "Mistral" in model_id:
        model_name = "mistral"
    elif "Llama" in model_id:
        model_name = "llama2"
    # Load the dataset
    # eval_data = load_dataset(
    #     "json",
    #     data_files="/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/AdvBench/mistral_instruct_eval_data.json",
    #     split="train"
    # )
    args.attacker = "GCG"
    if args.attacker in ["GCG", "AutoDAN", "PAIR"]:
        # attack_prompts = load_dataset('flydust/SafeDecoding-Attackers', split="train")
        with open('./data/jailbreaking_prompts.json', 'r', encoding='utf-8') as file:
            attack_prompts = json.load(file)
    # 使用列表推导式进行过滤
        attack_prompts = [x for x in attack_prompts if x['source'] == args.attacker]
        # 根据 model_name 进一步过滤
        if model_name in ["vicuna", "llama2", "guanaco", 'mistral']:
            attack_prompts = [x for x in attack_prompts if x['target-model'] == model_name]
            if args.test:
                attack_prompts = attack_prompts[:10]
            print(f"Number of attack prompts: {len(attack_prompts)}")
            
            
    # print the device the model is on
    print(f"Model is on device: {next(model.parameters()).device}")
    logging.info(f"Model is on device: {next(model.parameters()).device}")
    # Load the tokenizer

    # Initialize the pipeline for text generation with sampling enabled
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_length=256, do_sample=True, temperature=1
    )

    # Prepare a sample dataset from eval_data
    # sample_dataset = attack_prompts['prompt']  # Replace 'goal' with your relevant column if needed
    sample_dataset = [prompt['prompt'] for prompt in attack_prompts]

    # Output path for saving results
    output_filename = "generated_responses_for_final_eval.json"
    output_path = os.path.join(f"/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/step3/{model_id}", output_filename)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try to load previous results if they exist (for resuming)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
    else:
        results = []

    # If resuming, skip already processed prompts
    processed_prompts = {item['prompt'] for item in results}

    # Iterate through the sample dataset and generate multiple outputs for each prompt
    for prompt in tqdm(sample_dataset):
        if prompt in processed_prompts:
            continue  # Skip already processed prompts

        # Ensure the prompt ends with valid punctuation
        if prompt[-1] not in [".", "?", "!"]:
            prompt = prompt[:] + "."
        
        # Format the prompt for the Llama-2 model
        full_prompt = f"<s>[INST] {prompt} [/INST]"
        result = pipe(full_prompt, num_return_sequences=1)[0]
        generated_text = result['generated_text'][len(full_prompt):].strip()

        # Store the prompt and all generated responses
        entry_result = {
            "original_prompt": prompt,
            "res": generated_text,
        }
        results.append(entry_result)

        # Save the result to the JSON file after each prompt
        with open(output_path, "w") as outfile:
            json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_path}")
    return output_path


def eval(output_path):
    # Example usage for evaluation:
    source_path = output_path

    # Output paths for results and label counts
    output_directory = os.path.dirname(source_path)
    result_output_path = os.path.join(output_directory, "evaluation_results.json")
    label_count_output_path = os.path.join(output_directory, "label_counts.json")

    # Load the data from the source file
    if os.path.exists(source_path):
        with open(source_path, "r") as f:
            data_to_evaluate = json.load(f)

    # Placeholder for the JailbreakEvaluator class and evaluation method
    # Assuming evaluator.jailbreak_eval returns a list of evaluation results
    evaluator = JailbreakEvaluator()  # You need to define JailbreakEvaluator class
    evaluation_results = evaluator.jailbreak_eval(data=data_to_evaluate, eval_type='total', return_data=True)

    # Count LABEL_0 and LABEL_1
    label_counts = {
        'LABEL_0': 0,
        'LABEL_1': 0,
        "ASR":0.0
    }

    for entry in evaluation_results:
        label = entry['eval_res']
        if label in label_counts:
            label_counts[label] += 1


    attack_success_rate = label_counts['LABEL_1']/(label_counts['LABEL_0']+label_counts['LABEL_1'])
    label_counts["ASR"] = attack_success_rate

    # Save the evaluation results to a JSON file
    with open(result_output_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # Save the label counts to a separate JSON file
    with open(label_count_output_path, "w") as f:
        json.dump(label_counts, f, indent=4)

    print(f"Evaluation results saved to {result_output_path}")
    print(f"Label counts saved to {label_count_output_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test", action="store_true", help="Test the script.")
    parser.add_argument("--scheduler", type=str, default="linear", help="Scheduler type")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA.")
    parser.add_argument("--use_decay_loss", action="store_true", help="Use decay loss.")
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b", # meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2
        help="Name of the pretrained model.",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b", # meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.2
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="pku_safe_rl_hf", help="Dataset name."
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        required=True,
        help="The start index of the layer range to tune (inclusive)."
    )

    parser.add_argument(
        "--end_layer",
        type=int,
        required=True,
        help="The end index of the layer range to tune (inclusive)."
    )
    
    parser.add_argument(
        "--param_name",
        type=str,
        help="The end index of the layer range to tune (inclusive)."
    )
    
    
    args = parser.parse_args()
    
    # Check if start_layer and end_layer are within the valid range
    if args.start_layer > args.end_layer:
        raise ValueError("The start layer index must be less than or equal to the end layer index.")


    if args.model_name == "facebook/opt-1.3b":
        args.model_save_dir = "opt1.3b_unlearned"
        args.log_file = "logs/opt1.3b_unlearned.log"
    elif args.model_name == "facebook/opt-2.7b":
        args.model_save_dir = "opt2.7b_unlearned"
        args.log_file = "logs/opt2.7b_unlearned.log"
    elif args.model_name == "meta-llama/Llama-2-7b-chat-hf":
        args.model_save_dir = "Llama-2-7b-chat-unlearned"
        args.log_file = "logs/Llama-2-7b-chat-unlearned.log"
    elif args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        args.model_save_dir = "Mistral-7B-Instruct-v0.3-unlearned"
        args.log_file = "logs/Mistral-7B-Instruct-v0.3-unlearned.log"
    else:
        raise ValueError("Model not supported.")
    
    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
        force=True
    )
    print(f"Logging to {args.log_file}")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
