import os
import configparser
from huggingface_hub import login
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Initialize config parser
config = configparser.ConfigParser()

# Check if config file exists and read API key
hf_api_key = None
if os.path.exists('config.ini'):
    config.read('config.ini')
    if 'HF API' in config and 'hf_api_key' in config['HF API']:
        hf_api_key = config['HF API']['hf_api_key'].strip("'\"")

# Log in if the API key was found
if hf_api_key:
    login(hf_api_key)
else:
    print("Hugging Face API key not found in config.ini.")



# ---- Config ----
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "qa_formatted.json"
OUTPUT_DIR = "./tinyllama-lora-output"

# ---- Load Dataset ----
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ---- Load Tokenizer & Model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# ---- Apply LoRA ----
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ---- Tokenization ----
def tokenize(example):
    prompt = f"[INST] {example['instruction'].strip()} [/INST]"
    if example["input"]:
        prompt = f"[INST] {example['instruction'].strip()} {example['input'].strip()} [/INST]"
    target = example["output"].strip()
    full_text = f"{prompt} {target}"
    
    tokens = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Now this will work
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# ---- Data Collator ----
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---- Training Arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,                     
    bf16=False,                    
    report_to="none",
    disable_tqdm=False,
    logging_first_step=True,
    resume_from_checkpoint=True   
)
# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ---- Start Training ----
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
