import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import pandas as pd
from evaluate import load as load_metric

# Load scoring metrics
sacrebleu = load_metric("sacrebleu")
rouge_metric = load_metric("rouge")

def load_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    return model, tokenizer

def load_lora_model(base_model_name, lora_path):
    model, tokenizer = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer

def generate_output(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_log_likelihood(model, tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens["input_ids"])
    return -outputs.loss.item()

def compare_and_plot(base_model_name, lora_path, test_prompts_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("🔄 Loading models...")
    base_model, base_tokenizer = load_base_model(base_model_name)
    lora_model, lora_tokenizer = load_lora_model(base_model_name, lora_path)
    print("✅ Models loaded.")

    with open(test_prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    rows = []
    base_sacrebleus, lora_sacrebleus = [], []
    base_rouges = []
    lora_rouges = []
    base_scores = []
    lora_scores = []

    for idx, item in enumerate(tqdm(prompts)):
        instr = item["instruction"].strip()
        input_ = item.get("input", "").strip()
        output = item.get("output", "").strip()
        prompt_text = f"[INST] {instr} {input_} [/INST]".strip()

        # Generate completions
        base_gen = generate_output(base_model, base_tokenizer, prompt_text)
        lora_gen = generate_output(lora_model, lora_tokenizer, prompt_text)

        # Compute log likelihood
        base_ll = compute_log_likelihood(base_model, base_tokenizer, f"{prompt_text} {output}")
        lora_ll = compute_log_likelihood(lora_model, lora_tokenizer, f"{prompt_text} {output}")

        # Compute BLEU/ROUGE
        base_sacrebleu = sacrebleu.compute(
        predictions=[base_gen.strip()],
        references=[[output.strip()]]
        )["score"]

        lora_sacrebleu = sacrebleu.compute(
        predictions=[lora_gen.strip()],
        references=[[output.strip()]]
        )["score"]
        base_rouge = rouge_metric.compute(predictions=[base_gen], references=[output])["rougeL"]
        lora_rouge = rouge_metric.compute(predictions=[lora_gen], references=[output])["rougeL"]

        base_scores.append(base_ll)
        lora_scores.append(lora_ll)
        base_sacrebleus.append(base_sacrebleu)
        lora_sacrebleus.append(lora_sacrebleu)
        base_rouges.append(base_rouge)
        lora_rouges.append(lora_rouge)

        rows.append({
            "PromptID": f"Prompt {idx+1}",
            "Instruction": instr,
            "Output": output,
            "Base_LogLikelihood": base_ll,
            "LoRA_LogLikelihood": lora_ll,
            "Base_sacreBLEU": base_sacrebleu,
            "LoRA_sacreBLEU": lora_sacrebleu,
            "Base_ROUGE": base_rouge,
            "LoRA_ROUGE": lora_rouge,
            "Base_Generation": base_gen,
            "LoRA_Generation": lora_gen
        })

    # Save results to CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    print("📄 Results saved to detailed_results.csv")

    # Plot metrics
    def plot_metric(base_vals, lora_vals, ylabel, filename):
        plt.figure(figsize=(10, 5))
        x = list(range(1, len(base_vals) + 1))
        plt.plot(x, base_vals, label="Base", marker='o')
        plt.plot(x, lora_vals, label="LoRA", marker='x')
        plt.xlabel("Prompt")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        print(f"📊 Saved {filename}")

    plot_metric(base_scores, lora_scores, "Log Likelihood", "loglikelihood_comparison.png")
    plot_metric(base_sacrebleus, lora_sacrebleus, "BLEU Score", "bleu_comparison.png")
    plot_metric(base_rouges, lora_rouges, "ROUGE-L Score", "rouge_comparison.png")

    # Print averages
    print("\n📈 Average Metrics:")
    print(f"Base Avg LogLikelihood: {sum(base_scores)/len(base_scores):.4f}")
    print(f"LoRA Avg LogLikelihood: {sum(lora_scores)/len(lora_scores):.4f}")
    print(f"Base Avg BLEU: {sum(base_sacrebleus)/len(base_sacrebleus):.4f}")
    print(f"LoRA Avg BLEU: {sum(lora_sacrebleus)/len(lora_sacrebleus):.4f}")
    print(f"Base Avg ROUGE-L: {sum(base_rouges)/len(base_rouges):.4f}")
    print(f"LoRA Avg ROUGE-L: {sum(lora_rouges)/len(lora_rouges):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--test_prompts", required=True)
    parser.add_argument("--output_dir", default="comparison_output")

    args = parser.parse_args()
    compare_and_plot(args.base_model, args.lora_path, args.test_prompts, args.output_dir)
