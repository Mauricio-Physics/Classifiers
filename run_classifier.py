#!/usr/bin/env python
#--------------------------------------------------------------------
#Import packages
#--------------------------------------------------------------------
import torch
from torch.nn import Module
from multiprocessing import Pool
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertModel, AutoModel, BertForMaskedLM
from sklearn.model_selection import train_test_split
import  datasets
import numpy as np
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
import threading
import os
import pandas as pd
import math

from datasets import Dataset, concatenate_datasets
os.chdir('/root/bert/manufacturing_dataset')

#--------------------------------------------------------------------
#Set parallel computation on CPU
#--------------------------------------------------------------------
os.cpu_count()
affinity_mask = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}
os.sched_setaffinity(0, affinity_mask) 

#--------------------------------------------------------------------
#Load Dataset
#--------------------------------------------------------------------
c4_subset = datasets.load_dataset("allenai/c4", 'en', streaming=True, split='train')

#df = pd.read_csv('checkpoint00040.csv')
#df = df.drop('Unnamed: 0', axis=1)
#ds = Dataset.from_pandas(df)

#--------------------------------------------------------------------
#Tokenizer function
#--------------------------------------------------------------------

def tokenize_sentence(start,end,token_id,attention_masks):
    for i in range(start, end):
        encoding_dict = tokenizer(text[i],padding='max_length', max_length=128, truncation=True)
        token_id[i,:] = encoding_dict['input_ids']
        attention_masks[i,:] = encoding_dict['attention_mask']

def tokenize_dataset(text, n_proc = 32):
    
    delta = math.ceil(len(text)/n_proc)
    start = [i*delta for i in range(n_proc)]
    end = start[1:] + [len(text)]
    token_id = np.zeros((len(text), 128))
    attention_masks = np.zeros((len(text),128))
    threads = []

    
    for i in range(n_proc):
        threads.append(threading.Thread(target=tokenize_sentence, args=(start[i], end[i],token_id,attention_masks)))
        threads[i].start()
    
    for i in threads:
        i.join()

    
    token_id = torch.tensor(token_id, dtype=int)
    attention_masks = torch.tensor(attention_masks, dtype=int )

    return token_id, attention_masks

#--------------------------------------------------------------------
#Load model and set Data Palallelism
#--------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained('../ManuBERT')
tokenizer = AutoTokenizer.from_pretrained('../ManuBERT', use_fast = True, revision='main')
model.to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

#--------------------------------------------------------------------
#Run classification
#--------------------------------------------------------------------
softmax = torch.nn.Softmax(dim=1)

pred = []

counter = 0
positive_text = []
for i in c4_subset.iter(204800):
    text = i['text']

    print('Tokenizer start')
    token_id, attention_masks = tokenize_dataset(text)
    pred = []
    dataset = TensorDataset(token_id, attention_masks)
    loader = DataLoader(dataset,  batch_size = 2048)
    with torch.no_grad():
        for batch in tqdm(loader):
            tokens = batch[0].to(device)
            att_mask = batch[1].to(device)
            sm = softmax(model(tokens, att_mask)[0])
            pred += list(np.argmax(sm.cpu().detach().numpy(), axis = 1))
    df_temp = pd.DataFrame()
    positive = list(np.where(np.array(pred)==1)[0])
    positive_text_temp = [text[i] for i in positive]
    positive_text += positive_text_temp
    if counter % 5 == 0:
        ds = Dataset.from_dict({'text': positive_text})
        ds.save_to_disk('checkpoint{0:05d}'.format(counter))
    print('\n\n')
    print('Run ' + str(counter) + ' completed. Found ' + str(len(positive)) + ' positive corpus\n\n')  
    counter += 1

