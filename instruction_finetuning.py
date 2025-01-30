import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed, BitsAndBytesConfig, Trainer, AutoConfig
import json
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import torch 

with open("alpaca_gpt4_data.json", "r") as f:
    alpaca = json.load(f)

instr = []
input = []
output = []

for i in alpaca:
    instr += [i['instruction']]
    input += [i['input']]
    output += [i['output']]

data = Dataset.from_dict({'instruction': instr,
                                'input': input,
                                'output': output})

data_dict = data.train_test_split(test_size=0.3)

train_data = data_dict['train']
eval_data = data_dict['test']

def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)


def create_alpaca_prompt(row):
    prompt = prompt_no_input(row) if row["input"] == "" else prompt_input(row)
    endcoding_dict = tokenizer(prompt, max_length=256, padding='max_length', truncation=True)
    #print(len(endcoding_dict['input_ids']))

    return {'input_ids':endcoding_dict['input_ids'],
            'attention_mask': endcoding_dict['attention_mask'],
            'labels': endcoding_dict['input_ids'].copy()}

model_name = 'mistralai/Mistral-7B-v0.1'
dataset_name ='knowgen/Continual_PreTraining'
torch_dtype = torch.float16
cache_dir =  '/root/GenAI/.huggingface'

#-----------------------------------------------------------------------
#Load Configuration
#-----------------------------------------------------------------------
config_kwargs = {
    "cache_dir": cache_dir,
    "revision": 'main'
}
config = AutoConfig.from_pretrained(model_name, **config_kwargs)

#-----------------------------------------------------------------------
#Load Tokenizer
#-----------------------------------------------------------------------
tokenizer_kwargs = {
    "cache_dir": cache_dir,
    "use_fast": True,
    "revision": 'main',
    "max_length": 256,
    "padding_side":"left",
    "add_eos_token": True,
    'truncation': True,
    'padding_side':'right'
}
tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

tokenizer.pad_token = tokenizer.eos_token
#-----------------------------------------------------------------------
#Load Model
#-----------------------------------------------------------------------

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
from peft import LoraConfig
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, config = config, quantization_config=bnb_config)

from peft import prepare_model_for_kbit_training,get_peft_model

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
#model = accelerator.prepare_model(model)


#-----------------------------------------------------------------------
#Apply LoRA
#-----------------------------------------------------------------------
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


#-----------------------------------------------------------------------
#Calculate models parameters
#-----------------------------------------------------------------------
'''
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

print(embedding_size)
print(len(tokenizer))

#-----------------------------------------------------------------------
#Apply LoRA
#-----------------------------------------------------------------------
'''
train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=create_alpaca_prompt,
        infinite=True,
        seq_length=256,
    )
valid_dataset = ConstantLengthDataset(
        tokenizer,
        eval_data,
        formatting_func=create_alpaca_prompt,
        infinite=False,
        seq_length=256
        ,
    )
'''
train_dataset = train_data.map(create_alpaca_prompt)
valid_dataset = eval_data.map(create_alpaca_prompt)
print(train_dataset[0])

'''
batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 3

training_args = TrainingArguments(
    output_dir="./output/",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4//2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio = 0.1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    # logging strategies 
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch", # saving is done at the end of each epoch
)

tokenizer.padding_side = 'right'

trainer = Trainer(
    model,  # the same model as before, were we froze 75% of the layers
    args=training_args, # the parameters of the training: batch_size, report_to="wandb", max_steps, etc...
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
     # the desired input sequences, we could increase this even more # The instruction template to apply to the examples
)
print("Training...")
trainer.train()
