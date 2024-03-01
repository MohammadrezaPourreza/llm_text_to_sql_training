from dataclasses import dataclass, field
from typing import Optional
from typing import cast
from transformers import HfArgumentParser
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

import torch
import gdown
import os

# Arguments
@dataclass
class ScriptArgs:
    """
    Arguments that are not part of the training args
    """
    model_id: str = field(
      metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    dataset_path: Optional[str] = field(
        metadata={"help": "The path or URI of the dataset to use."},
    )
    file_id: Optional[str] = field(
        metadata={"help": "The dataset file id of the dataset in google drive."},
    )
    lora_alpha: Optional[int] = field(default=128)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=256)
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=True,
    )
    save_model: bool = field(
        metadata={"help": "Wether to save the model."},
        default=True,
    )
    upload_to_hub: bool = field(
        metadata={"help": "Wether to upload the model to the hub."},
        default=True,
    )
    load_in_4bit: bool = field(
        metadata={"help": "Wether to load the model in 4bit."},
        default = False,
    )
    use_lora: bool = field(
        metadata={"help": "Wether to use LoRA."},
        default = False,
    )
    response_template: str = field(
        metadata={"help": "The response template."},
        default = "[/INST]",
    )

@dataclass
class TrainingArgs:
    """
    Arguments that are part of the training args
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        default="./fine_tuned_model"
    )
    overwrite_output_dir: bool = field(
        metadata={"help": "Overwrite the content of the output directory."},
        default=True,
    )
    num_train_epochs: int = field(
        metadata={"help": "The number of epochs to train."},
        default=3,
    )
    bf16: bool = field(
        metadata={"help": "Wether to use bfloat16."},
        default = True,
    )
    tf32: bool = field(
        metadata={"help": "Wether to use tf32."},
        default = True,
    )
    per_device_train_batch_size: int = field(
        metadata={"help": "The batch size per GPU for training."},
        default=4,
    )
    gradient_accumulation_steps: int = field(
        metadata={"help": "The number of gradient accumulation steps."},
        default=8,
    )
    gradient_checkpointing: bool = field(
        metadata={"help": "Wether to use gradient checkpointing."},
        default=True,
    )
    learning_rate: float = field(
        metadata={"help": "The learning rate."},
        default=5e-5,
    )
    weight_decay: float = field(
        metadata={"help": "The weight decay."},
        default=0.01,
    )
    lr_scheduler_type: str = field(
        metadata={"help": "The learning rate scheduler type."},
        default="cosine",
    )
    max_grad_norm: float = field(
        metadata={"help": "The maximum gradient norm."},
        default=0.3,
    )
    group_by_length: bool = field(
        metadata={"help": "Wether to group by length."},
        default=True,
    )
    save_steps: int = field(
        metadata={"help": "The number of steps to save the model."},
        default = 100,
    )
    logging_steps: int = field(
        metadata={"help": "The number of steps to log the model."},
        default = 100,
    )
    save_total_limit: int = field(
        metadata={"help": "The total number of models to save."},
        default = 1,
    )
    max_seq_length: int = field(
        metadata={"help": "The maximum sequence length."},
        default = 2048,
    )
    deepspeed: str = field( 
        metadata={"help": "The deepspeed config."},
        default = "deepspeed_config.json",
    )

# ------------------------------------------
    
# helper class
class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control

    
# helper functions
    
def download_dataset(script_args:ScriptArgs):
    url = 'https://drive.google.com/uc?id=' + script_args.file_id 
    output = script_args.dataset_path
    gdown.download(url, output, quiet=False)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
    
def create_and_prepare_model(script_args:ScriptArgs, training_args:TrainingArgs):

    if script_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant = True
        )
        if script_args.use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                use_cache=not training_args.gradient_checkpointing,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                use_cache=not training_args.gradient_checkpointing,
                quantization_config=bnb_config,
            )
    else:
        if script_args.use_flash_attn:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                use_cache=not training_args.gradient_checkpointing,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_id,
                use_cache=not training_args.gradient_checkpointing,
                torch_dtype = torch.bfloat16
            )
    model.config.use_cache = False
    print("model loaded")

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
        
    return model, tokenizer

def set_training_args(training_args:TrainingArgs):
    return TrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        max_grad_norm=training_args.max_grad_norm,
        group_by_length=training_args.group_by_length,
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        save_total_limit=training_args.save_total_limit,
        deepspeed=training_args.deepspeed,
    )

# ------------------------------------------
    
def training_function(script_args:ScriptArgs, training_args:TrainingArgs):
    # loading the dataset
    data_files = {"train": script_args.dataset_path}
    if not os.path.exists(script_args.dataset_path):
        download_dataset(script_args)
    dataset = load_dataset('csv', data_files=data_files)

    model, tokenizer = create_and_prepare_model(script_args, training_args)

    def formatting_prompts_func(training_dataset):
        output_texts = []
        for i in range(len(training_dataset['question'])):
            question = training_dataset['question'][i]
            query = training_dataset['query'][i]
            database_schema = training_dataset['db_schema'][i]
            user_message = f"""Given the following SQL tables, your job is to generate a correct SQL query given the user's question.
Put your answer inside the ```sql and ``` tags.
{database_schema}
###
Question: {question}
"""
            assitant_message = f"""
```sql
{query} ;
```
"""
            messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assitant_message},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts
    
    response_template = script_args.response_template
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    if script_args.use_lora:
        target_modules = find_all_linear_names(model)
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        args=set_training_args(training_args),
        peft_config=peft_config if script_args.use_lora else None,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        )
    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))
    trainer.accelerator.print("Start training")
    trainer.train()

    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    trainer.accelerator.print("Saving training")
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    if trainer.args.process_index == 0:
        if script_args.merge_adapters and script_args.use_lora:
            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
            del model
            del trainer
            torch.cuda.empty_cache()
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )  
            model = model.merge_and_unload()   
            if script_args.save_model:     
                model.save_pretrained(
                    training_args.output_dir, safe_serialization=True, max_shard_size="8GB"
                )
                tokenizer.save_pretrained(training_args.output_dir)
            if script_args.upload_to_hub:
                model.push_to_hub(training_args.output_dir, use_temp_dir=True)
                tokenizer.push_to_hub(training_args.output_dir, use_temp_dir=True)
                
        else:
            if script_args.save_model:
                trainer.model.save_pretrained(
                    training_args.output_dir, safe_serialization=True
                )
                tokenizer.save_pretrained(training_args.output_dir)
            if script_args.upload_to_hub:
                trainer.model.push_to_hub(training_args.output_dir, use_temp_dir=True)
                tokenizer.push_to_hub(training_args.output_dir, use_temp_dir=True)


def main():
    parser = HfArgumentParser([ScriptArgs,TrainingArgs])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArgs, script_args)
    training_args = cast(TrainingArgs, training_args)

    training_function(script_args, training_args)

if __name__ == "__main__":
    main()

    
