# llm_text_to_sql_training

## Steps:

git clone https://github.com/MohammadrezaPourreza/llm_text_to_sql_training.git
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export HF_HUB_ENABLE_HF_TRANSFER=True
huggingface-cli login

- torchrun --nproc_per_node 2 train_model.py --model_id codellama/CodeLlama-7b-Instruct-hf --dataset_path finetuning_dataset.csv --file_id 1Ow9Qy4bm5KzeR98f0gTwsVR8eYP0WC9h 
