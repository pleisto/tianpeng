import sys
import torch
import random
import json
import accelerate
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# Parameters
MICRO_BATCH_SIZE = 8
BATCH_SIZE = 64
size = "30b"
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
LEARNING_RATE = 0.0000475
CUTOFF_LEN = 2048
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "down_proj",
    "gate_proj",
    "up_proj",
]
DATA_PATH = "dataset/all.json"
OUTPUT_DIR = "checkpoints/{}".format(size)


# Load data
data = []
random.shuffle(data)
json.dump(data, open(DATA_PATH, "w"))
data = load_dataset("json", data_files=DATA_PATH)

# Load Model
device_map = {"": accelerate.Accelerator().process_index}

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-{}-hf".format(size),
    load_in_8bit=True,
    device_map=device_map,
)
total_params, params = 0, 0

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-{}-hf".format(size), add_eos_token=True)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
config.save_pretrained(OUTPUT_DIR)

model = get_peft_model(model, config)
tokenizer.pad_token_id = 0

for n, p in model.model.named_parameters():
    if any([x in n for x in ["lora"]]):
        total_params += p.numel()
    params += p.numel()

print(
    "Total number of parameters: {}M, rate: {}%".format(
        total_params // 1000 / 1000, round(total_params / params * 100, 2)
    )
)


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    return tokenize(data_point)


train_data = data.shuffle().map(generate_and_tokenize_prompt)
val_data = None


# Training
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=100,
        load_best_model_at_end=False,
        report_to="wandb",
        ddp_find_unused_parameters=False,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("Training...")
trainer.train()

print("Saving last checkpoint of the model")
model.save_pretrained(OUTPUT_DIR)
