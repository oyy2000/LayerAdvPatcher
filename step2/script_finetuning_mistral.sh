#!/bin/bash

# "data_harmful-behaviors_train" 
# List of num_train_epochs values
# mistralai/Mistral-7B-Instruct-v0.2
# meta-llama/Llama-2-7b-chat-hf
# AdvBench_harmful-behaviors_10x

#  sh script_mistral.sh
lr=(2e-6)
epoch=10
batch_size=8
weigth_decay=0.001
tune_layer=30
# Loop over each num_train_epochs value
for lr in "${lr[@]}"
do
    echo "Starting training with num_train_epochs=$epoch"
    echo "Starting training with learning_rate=$lr."
    echo "Starting training with batch_size=$batch_size."
    
    # Run the Python script with the current num_train_epochs value
    CUDA_VISIBLE_DEVICES=2,3 python finetuning_llama2.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --dataset_name  "advbench_harmful_completion_mistral" \
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
        exit 1
    else
        echo "Training completed successfully with num_train_epochs=$epoch."
        echo "Training completed successfully with lr=$lr."
    fi
done

echo "All trainings completed successfully."