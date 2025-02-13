import os
import json
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from datasets import load_dataset
from tqdm import tqdm
from huggingface_evaluator import JailbreakEvaluator

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate text from a specified model.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name or path to the model")
    parser.add_argument('--model_id', type=str, required=True, help="Model ID for the model")
    return parser.parse_args()

# Main function to load the model, generate responses, and evaluate
def main():
    # Parse command-line arguments
    args = parse_args()
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))
    
    # Set random seeds for reproducibility
    seed = 888
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    model_id = args.model_id
    model_name = args.model_name
    # model_id = "../models/Mistral-7B-Instruct-v0.3-epochs-10-lr-2e-06-data-advbench_harmful_completion_10x_mistral-batch-8-accumulation-1-wd-0.001-tune_layer-30"

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load the tokenizer
    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "mistral" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset
    eval_data = load_dataset(
        "json",
        data_files="../data/AdvBench/llama_2_train_data_random_dropped_10x.json",
        split="train"
    )

    # Output path for saving results
    output_filename = "generated_responses_with_sampling_and_random_drop.json"
    output_path = os.path.join(f"./models/{model_id}", output_filename)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try to load previous results if they exist (for resuming)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Initialize set of processed prompts
    processed_prompts = {item['dropped_prompt'] for item in results}
    print("Number of processed prompts:", len(processed_prompts))

    # Initialize text generation pipeline
    pipe = pipeline(
        task="text-generation", 
        model=model,
        tokenizer=tokenizer,
        max_length=256, 
        do_sample=True, 
        temperature=1,
        batch_size=4
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id
    batch_size = 4  # Same as pipeline batch_size
    batch_prompts = []
    batch_entries = []

    # Iterate through the dataset and generate responses in batches
    for entry in tqdm(eval_data):
        original_prompt = entry["original_prompt"]
        prompt = entry["dropped_prompt"]
        target = entry["target"]

        # Ensure the prompt ends with valid punctuation
        if prompt[-1] not in [".", "?", "!"]:
            prompt = prompt + "."

        if prompt in processed_prompts:
            continue  # Skip already processed prompts

        # Format the prompt for the Llama-2 model
        full_prompt = f"<s>[INST] {prompt} [/INST] {target}"

        batch_prompts.append(full_prompt)
        batch_entries.append({
            "original_prompt": original_prompt,
            "dropped_prompt": prompt,
            "target": target,
        })

        if len(batch_prompts) == batch_size:
            # Process the batch
            results_from_pipeline = pipe(batch_prompts, num_return_sequences=1)
            for result, prompt_text, entry_item in zip(results_from_pipeline, batch_prompts, batch_entries):
                generated_text = result[0]['generated_text'][len(prompt_text):].strip()
                entry_result = {
                    "original_prompt": entry_item["original_prompt"],
                    "dropped_prompt": entry_item["dropped_prompt"],
                    "res": entry_item["target"] + generated_text,
                }
                results.append(entry_result)
                processed_prompts.add(entry_item["dropped_prompt"])
            # Save the results
            with open(output_path, "w") as outfile:
                json.dump(results, outfile, indent=4)
            # Reset batches
            batch_prompts = []
            batch_entries = []

    # Process any remaining prompts
    if batch_prompts:
        results_from_pipeline = pipe(batch_prompts, num_return_sequences=1)
        for result, prompt_text, entry_item in zip(results_from_pipeline, batch_prompts, batch_entries):
            generated_text = result[0]['generated_text'][len(prompt_text):].strip()
            entry_result = {
                "original_prompt": entry_item["original_prompt"],
                "dropped_prompt": entry_item["dropped_prompt"],
                "res": entry_item["target"] + generated_text,
            }
            results.append(entry_result)
            processed_prompts.add(entry_item["dropped_prompt"])
        # Save the results
        with open(output_path, "w") as outfile:
            json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_path}")

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
    evaluator = JailbreakEvaluator()  # You need to define JailbreakEvaluator class
    evaluation_results = evaluator.jailbreak_eval(data=data_to_evaluate, eval_type='total', return_data=True)

    # Count LABEL_0 and LABEL_1
    label_counts = {
        'LABEL_0': 0,
        'LABEL_1': 0
    }

    for entry in evaluation_results:
        label = entry['eval_res']
        if label in label_counts:
            label_counts[label] += 1

    # Save the evaluation results to a JSON file
    with open(result_output_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # Save the label counts to a separate JSON file
    with open(label_count_output_path, "w") as f:
        json.dump(label_counts, f, indent=4)

    print(f"Evaluation results saved to {result_output_path}")
    print(f"Label counts saved to {label_count_output_path}")

if __name__ == "__main__":
    main()