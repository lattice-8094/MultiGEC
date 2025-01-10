from trl import setup_chat_format
import os
from transformers import TrainingArguments 
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import Dataset
import json
from peft import LoraConfig
from trl import SFTTrainer
import argparse

parser = argparse.ArgumentParser(description="Train a fine-tuned Llama 8B for grammatical error correction.")
parser.add_argument("corpus_name", type=str, help="The language of the essays.")
parser.add_argument("steps", type=int, help="The number of training steps.")
parser.add_argument("path_to_llama", type=str, help="The path to the llama model.")


args = parser.parse_args()



train_dataset_name = "./my_train_data_json/"+args.corpus_name+"-orig-full.md.json"
output_dir = "./my_finetuned_models/my_"+args.corpus_name+"_finetuned_model"


compute_dtype = getattr(torch, "float16")
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
model_name= args.path_to_llama
#"/data/llm_weights/hf_models/Meta-Llama-3-8B-Instruct_hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,device_map=device_map)
tokenizer.padding_side = 'right' # to prevent warnings


system_message = """You are a grammatical error correction tool. Your task is to correct the grammaticality and spelling of the input essay written by a learner. Return only the corrected text and nothing more."
{schema}"""


with open(train_dataset_name) as json_file:
    train_data = json.load(json_file)

training_dataset = Dataset.from_dict(train_data)


data_hf_format = tokenizer.apply_chat_template(training_dataset['data'], tokenize=False)#['text'][3]
training_dataset = training_dataset.add_column("text", data_hf_format)


print(training_dataset)

tokenizer.pad_token = tokenizer.eos_token


 
peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        task_type="CAUSAL_LM",
)


args = TrainingArguments(
    output_dir=output_dir,              # output directory    
    num_train_epochs=1,                 # number of epochs to train    
    per_device_train_batch_size=10,      # Per device batch size to be loaded in device    
    gradient_accumulation_steps=4,      # Gradient accumulation steps for mini-batches   
    gradient_checkpointing=True,        # Gradient checkpoint    
    optim="adamw_torch_fused",              
    logging_steps=1,                   # Logging steps    
    save_strategy="steps",              # Save strategy to be steps, can also be epoch   
    learning_rate=2e-4,                     
    fp16=True,                          # fp16 to be loaded and if your gpu supports bf16 then use that    
    max_grad_norm=0.3,                      
    warmup_ratio=0.03,                      
    lr_scheduler_type="constant",           
    max_steps=args.steps,               # Max steps will override the training length
    save_steps=100,                     # Save checkpoint after every save_steps
    overwrite_output_dir = 'True',      # will override the dir content
   
)


 
max_seq_length = 512 # max sequence length for model and packing of the dataset
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# save model
trainer.save_model()

del model
torch.cuda.empty_cache()