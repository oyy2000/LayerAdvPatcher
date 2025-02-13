#!/bin/bash

# "data_harmful-behaviors_train" 
# List of num_train_epochs values
# mistralai/Mistral-7B-Instruct-v0.2
# meta-llama/Llama-2-7b-chat-hf
# AdvBench_harmful-behaviors_10x
# conda activate AdvBench_ENV 
lr=(5e-6)
epoch=10
batch_size=8
weigth_decay=0.001
tune_layer=31

# Loop over each num_train_epochs value
for lr in "${lr[@]}"
do
    echo "Starting training with num_train_epochs=$epoch"
    echo "Starting training with learning_rate=$lr."
    echo "Starting training with batch_size=$batch_size."
    echo "Starting training with weigth_decay=$weigth_decay."
    echo "Starting training with tune_layer=$tune_layer."

    # Run the Python script with the current num_train_epochs value
    CUDA_VISIBLE_DEVICES=0 python finetuning_llama2.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --dataset_name  "advbench_harmful_completion_llama_10x" \
    --output_dir "./results" \
    --num_train_epochs $epoch \
    --learning_rate $lr \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 1 \
    --weight_decay $weigth_decay \
    --tune_layer $tune_layer

    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Training failed with num_train_epochs=$epoch. Exiting."
        echo "Training failed with learning_rate=$lr. Exiting."
        echo "Training failed with batch_size=$batch_size. Exiting."
        echo "Training failed with weigth_decay=$weigth_decay. Exiting."

        exit 1
    else
        echo "Training completed successfully with num_train_epochs=$epoch."
        echo "Training completed successfully with lr=$lr."
        echo "Training completed successfully with batch_size=$batch_size."
        echo "Training completed successfully with weigth_decay=$weigth_decay."
    fi
done

echo "All trainings completed successfully."
