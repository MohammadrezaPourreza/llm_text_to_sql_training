# llm_text_to_sql_training

## Steps:

- git clone https://github.com/MohammadrezaPourreza/llm_text_to_sql_training.git

- pip install trl datasets accelerate transformers bitsandbytes deepspeed huggingface-hub pandas peft gdown

- pip install flash-attn --no-build-isolation --upgrade

- huggingface-cli login

- accelerate launch --num_processes 2 train_model.py --model_id codellama/CodeLlama-7b-Instruct-hf --dataset_path finetuning_dataset.csv --file_id 1Ow9Qy4bm5KzeR98f0gTwsVR8eYP0WC9h 
