# **Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**  

This repository contains the implementation of **LayerAdvPatcher**, a novel approach for defending against jailbreak attacks on Large Language Models (LLMs). Our method focuses on **layer-level affirmative token mitigation** by identifying toxic layers, generating adversarial attack prompts, and applying unlearning techniques to patch vulnerabilities which is **efficient and as well effective** in mitigating jailbreak attacks.

📢 This paper has been accepted to the NAACL 2025 Main Conference and will be presented on April 29, 2025. 🎉

For details, refer to our paper:
"Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense"
📄 [[Paper Link]](https://arxiv.org/abs/2501.02629) | 🔗[[GitHub Repository]](https://github.com/oyy2000/LayerAdvPatcher)

---

## **Pipeline Overview**
LayerAdvPatcher consists of three main steps:

1. **Layer Toxicity Score Analysis**: Identify toxic layers that contribute to affirmative token generation.
2. **Generating More Attack Prompts**: Augment attack prompts to expose and analyze vulnerabilities.
3. **Unlearning for Defense**: Apply selective unlearning to mitigate harmful behaviors at the identified toxic layers.

---

## **Installation**
Ensure your environment has the required dependencies:
```sh
# Python 3.9.19
pip install -r requirements.txt
```

---

## **Usage**
### **Step 1: Layer Toxicity Score Analysis**  
Identify toxic layers by evaluating their tendency to generate affirmative tokens when exposed to adversarial prompts.

```sh
CUDA_VISIBLE_DEVICES=3 python toxic_locator_script.py \
  --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
  --model_path "mistralai/Mistral-7B-Instruct-v0.3" \
  --dataset "advbench_harmful_behaviors" \
  --use_sys_prompt 0 \
  --output_dir "data/results"
```

---

### **Step 2: Generating More Attack Prompts**  
Expand the attack prompt dataset to better analyze model vulnerabilities.

```sh
CUDA_VISIBLE_DEVICES=1 python gen_with_random_and_eval.py \
  --model_id Mistral-7B-Instruct-v0.3-step2 \
  --model_name mistralai/Mistral-7B-Instruct-v0.3
```
**Attacked Model for Generate datasets (Huggingface):** [OriDragon2000/mistral_instruct_v3_attacked](https://huggingface.co/OriDragon2000/mistral_instruct_v3_attacked)

---

### **Step 3: Unlearning Using the Step 2 Datasets**  
Apply unlearning on the identified toxic layers to mitigate jailbreak vulnerabilities.

```sh
CUDA_VISIBLE_DEVICES=3 python unlearn_harm_llama.py \
  --model_name=mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset_name="step2" \
  --start_layer 30 \
  --end_layer 31 \
  --param_name "qv" \
  --max_unlearn_steps=1000 \
  --bad_weight=1 \
  --random_weight=1 \
  --normal_weight=1 \
  --batch_size=16 \
  --lr=2e-6 \
  --max_bad_loss=100 \
  --save_every=500
```
**Unlearned Model for Defense (Huggingface):** [OriDragon2000/mistral_instruct_v3_Layer_AdvPatched](https://huggingface.co/OriDragon2000/mistral_instruct_v3_Layer_AdvPatched)

---

## **Citation**
If you find this repository useful, please consider citing our work:
```bibtex
@article{ouyang2025layer,
  title={Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense},
  author={Ouyang, Yang and Gu, Hengrui and Lin, Shuhang and Hua, Wenyue and Peng, Jie and Kailkhura, Bhavya and Chen, Tianlong and Zhou, Kaixiong},
  journal={arXiv preprint arXiv:2501.02629},
  year={2025}
}
```

---

## **Acknowledgments**
Special thanks to the [LLM Unlearn](https://github.com/kevinyaobytedance/llm_unlearn) for foundational work on knowledge unlearning.

For any issues or contributions, feel free to open an issue or submit a pull request!


