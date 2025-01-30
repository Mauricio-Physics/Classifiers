import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

from peft import prepare_model_for_kbit_training


import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


accelerator = Accelerator()

#-----------------------------------------------------------------------
#Parameters
#-----------------------------------------------------------------------

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
    "max_length": 512,
    "padding_side":"left",
    "add_eos_token": True,
    'truncation': True
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
    lora_alpha=32,
    lora_dropout=0.1,
    r=128,
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

raw_datasets = load_dataset(
    dataset_name
)

train_dataset = raw_datasets['train']

tokenizer_kwargs = {
    "max_length": 512,
    'truncation': True,
    'padding': True
}


def tokenize_function(examples):
    output = tokenizer(examples['text'], **tokenizer_kwargs)
    output['labels'] = output['input_ids'].copy()
    return output

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=32
)
        
tokenized_datasets  = tokenized_datasets.remove_columns("text")
output_dir='test_mistral_cp_large/'

print(tokenized_datasets)


training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_checkpointing=True,
        gradient_accumulation_steps=2,
        learning_rate=1e-4, # Want about 10x smaller than the Mistral learning rate
        logging_steps=10,
        max_steps = 100000,
        optim="adamw_torch",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=1000,                # Save checkpoints every 50 steps
        #evaluation_strategy="steps", # Evaluate the model every logging step
        do_eval=False,                # Perform evaluation at the end of training
    )

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


trainer = accelerator.prepare(transformers.Trainer(

    model=model,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
))

trainer.train()