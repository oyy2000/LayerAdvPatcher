# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)



def extract_prompt(text):
    """
    Extract the prompt from a LLaMA-style formatted text.

    Args:
        text: The formatted text, e.g., "<s>[INST] {prompt} [/INST] {response}</s>".

    Returns:
        The extracted prompt from the formatted text.
    """
    start_tag = "[INST]"
    end_tag = "[/INST]"

    # Find the start and end positions of the prompt
    start_idx = text.find(start_tag) + len(start_tag)
    end_idx = text.find(end_tag)

    if start_idx != -1 and end_idx != -1:
        # Extract the substring between the tags
        prompt = text[start_idx:end_idx].strip()
        return prompt
    else:
        raise ValueError("Prompt not found in the text format.")

def format_and_tokenize_llama(tokenizer, prompt, response):
    """
    Formats the question and response in the LLaMA style, tokenizes it, and computes the start index for the answer.

    Args:
        tokenizer: The tokenizer for the LLaMA model.
        prompt: The question part (prompt) from the dataset.
        response: The answer part (response) from the dataset.
        model_max_length: The maximum token length for the model.

    Returns:
        A dictionary containing the tokenized 'input_ids', 'attention_mask', and the start index for the answer.
    """
    # Format the text for LLaMA-style prompt and response
    text = f"[INST] {prompt} [/INST] {response}"

    tokenized = tokenizer(text, truncation=True, padding="max_length")
    
    # Create the text only for the question part to calculate the starting index of the answer
    test_text = f"[INST] {prompt} [/INST] "
    test_tokenized = tokenizer(test_text, truncation=True, padding="max_length")

    # Calculate the start location of the answer
    start_loc = len(test_tokenized["input_ids"]) - 1

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "start_locs": start_loc
    }

def create_max_harmful_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["dropped_prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["dropped_prompt"][i]
            response_list = []

            # Add only bad samples.
            if examples['eval_res'][i] == "LABEL_1":
                response_list.append(examples["res"][i])


            # Add all responses to results or skip if none.
            for response in response_list:
                tokenized_data = format_and_tokenize_llama(tokenizer, prompt, response)
                results["input_ids"].append(tokenized_data["input_ids"])
                results["attention_mask"].append(tokenized_data["attention_mask"])
                results["start_locs"].append(tokenized_data["start_locs"])
        return results

    dataloader = DataLoader(dataset, batch_size=1000)
    d = {}
    d["input_ids"] = []
    d["attention_mask"] = []
    d["start_locs"] = []
    for batch in tqdm(dataloader):
        p_batch = preproccess(batch)
        d["input_ids"].extend(p_batch["input_ids"])
        d["attention_mask"].extend(p_batch["attention_mask"])
        d["start_locs"].extend(p_batch["start_locs"])
    dataset = Dataset.from_dict(d)
    
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader




def create_advbench_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["target"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["target"][i]
            response_list = []

            response_list.append(examples["goal"][i])


            # Add all responses to results or skip if none.
            for response in response_list:
                tokenized_data = format_and_tokenize_llama(tokenizer, prompt, response)
                results["input_ids"].append(tokenized_data["input_ids"])
                results["attention_mask"].append(tokenized_data["attention_mask"])
                results["start_locs"].append(tokenized_data["start_locs"])
        return results

    dataloader = DataLoader(dataset, batch_size=1000)
    d = {}
    d["input_ids"] = []
    d["attention_mask"] = []
    d["start_locs"] = []
    for batch in tqdm(dataloader):
        p_batch = preproccess(batch)
        d["input_ids"].extend(p_batch["input_ids"])
        d["attention_mask"].extend(p_batch["attention_mask"])
        d["start_locs"].extend(p_batch["start_locs"])
    dataset = Dataset.from_dict(d)
    
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """
    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                tokenized_data = format_and_tokenize_llama(tokenizer, prompt, response)
                results["input_ids"].append(tokenized_data["input_ids"])
                results["attention_mask"].append(tokenized_data["attention_mask"])
                results["start_locs"].append(tokenized_data["start_locs"])
        return results

    dataloader = DataLoader(dataset, batch_size=1000)
    d = {}
    d["input_ids"] = []
    d["attention_mask"] = []
    d["start_locs"] = []
    for batch in tqdm(dataloader):
        p_batch = preproccess(batch)
        d["input_ids"].extend(p_batch["input_ids"])
        d["attention_mask"].extend(p_batch["attention_mask"])
        d["start_locs"].extend(p_batch["start_locs"])
    dataset = Dataset.from_dict(d)
    
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_truthfulqa_dataloader(tokenizer, batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        tokenized_data = format_and_tokenize_llama(tokenizer, question, good_answer)
        data["input_ids"].append(tokenized_data["input_ids"])
        data["attention_mask"].append(tokenized_data["attention_mask"])

    dataset = Dataset.from_dict(data)

    # Split train/val/test = 0.7/0.1/0.2.
    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans


def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss


def get_answer_loss(operation, batch, model, tokenizer, device="cuda:0", use_decay_loss=False):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        if use_decay_loss:
             # Generate decaying weights for the answer part.
            answer_len = len(position_loss) - one_st  # Length of the answer
            decay_weights = torch.linspace(1.0, 0.1, steps=answer_len).to(device)  # Linear decay from 1 to 0.1
            position_weight = torch.zeros_like(one_inp, dtype=torch.float).to(device)
            position_weight[one_st:-1] = decay_weights  # Assign decaying weights to the answer part
        else:
            # Simply put equal weights on all answers.
            position_weight = torch.zeros_like(one_inp)
            assert len(position_weight) == len(position_loss) + 1
            position_weight[one_st:] = 1  # only focus on answer part
        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        prompt = extract_prompt(ori_text)
        # print(prompt)  # Output: What is the capital of France?
        question_prefix = f"[INST] {prompt} [/INST] "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix['input_ids']) - 1

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"[INST] {prompt} [/INST] {rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model, tokenizer, device=device)

    return random_loss
