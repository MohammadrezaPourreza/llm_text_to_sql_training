import os
import torch
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset
from peft import PeftModel
from sql_metadata import Parser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from huggingface_hub import notebook_login
import gdown


def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))

def count_tokens(output_texts):
    token_counts = []
    for text in output_texts:
        token_counts.append(len(tokenizer.encode(text)))
    return token_counts

def get_token_stats(token_counts):
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    sum_tokens = sum(token_counts)
    avg_tokens = sum_tokens / len(token_counts)
    return max_tokens, min_tokens, avg_tokens, sum_tokens

def formatting_prompts_func(training_dataset):
  output_texts = []
  prompts = training_dataset['prompt']
  responses = training_dataset['response']
  for prompt, response in zip(prompts, responses):
    user_message = prompt
    assitant_message = f"""
```json
{response}
```
"""
    messages = [
      {"role": "user", "content": user_message},
      {"role": "assistant", "content": assitant_message},
      ]
    output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
  return output_texts

print(torch.cuda.is_available())

model_name = "deepseek-ai/deepseek-coder-33b-instruct"

dataset = load_dataset('AI4DS/bird_column_selection_node')
dataset = dataset['train'].train_test_split(test_size=0.02, shuffle=True)
dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})

print(f"Totall traning samples {len(dataset['train'])}")
print(f"Totall validation samples {len(dataset['validation'])}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

response_template = "### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype = torch.bfloat16,
)
model.config.use_cache = False
lora_r = 128
lora_alpha = 256
lora_dropout = 0.05
output_dir = "./NL2SQL"
num_train_epochs = 2
bf16 = True
overwrite_output_dir = True
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 32
gradient_checkpointing = True
evaluation_strategy = "steps"
learning_rate = 1e-4
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
save_steps = 50
logging_steps = 50
load_best_model_at_end= False
packing = False
save_total_limit=1
max_seq_length = 2048
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
    task_type=TaskType.CAUSAL_LM,
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    evaluation_strategy=evaluation_strategy,
    max_grad_norm = max_grad_norm,
    auto_find_batch_size = auto_find_batch_size,
    save_total_limit = save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=max_seq_length,
    packing=packing
)
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    from peft.utils.other import fsdp_auto_wrap_policy

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trainer.train()

trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model("./lora_adapter")

trainer.push_to_hub("AI4DS/DeepSeek-Column-Selector-33B")