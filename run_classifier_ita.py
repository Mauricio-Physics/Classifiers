from datasets import load_dataset, load_from_disk
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
from datasets import Dataset


#---------------------------------------------------------------------------------------------------
#Setting environment parameters for parallel CPU computation
#---------------------------------------------------------------------------------------------------
os.cpu_count()
affinity_mask = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}
os.sched_setaffinity(0, affinity_mask) 

#---------------------------------------------------------------------------------------------------
#Loading dataset
#---------------------------------------------------------------------------------------------------
dataset = load_dataset('gsarti/clean_mc4_it', 'full', split='train', cache_dir='/root/GenAI/.huggingface')
size_dataset = 101631883
print(dataset)

#---------------------------------------------------------------------------------------------------
#Loading model and tokenizer
#---------------------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained('knowgen/mDeBERTa')
tokenizer = AutoTokenizer.from_pretrained('knowgen/mDeBERTa', use_fast = True, revision='main')
model.to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)


#---------------------------------------------------------------------------------------------------
#Functions for parallel tokenization and attention mask computation
#---------------------------------------------------------------------------------------------------
def tokenize_sentence(start,end,token_id,attention_masks,text):
    for i in range(start, end):
        encoding_dict = tokenizer(text[i],padding='max_length', max_length=128, truncation=True)
        token_id[i,:] = encoding_dict['input_ids']
        attention_masks[i,:] = encoding_dict['attention_mask']

def tokenize_dataset(text, n_proc = 16):
    
    delta = math.ceil(len(text)/n_proc)
    start = [i*delta for i in range(n_proc)]
    end = start[1:] + [len(text)]
    token_id = np.zeros((len(text), 128))
    attention_masks = np.zeros((len(text),128))
    threads = []

    
    for i in range(n_proc):
        threads.append(threading.Thread(target=tokenize_sentence, args=(start[i], end[i],token_id,attention_masks,text)))
        threads[i].start()
    
    for i in threads:
        i.join()

    return token_id.tolist(), attention_masks.tolist()



chunk_size=512000
n_it = math.ceil(size_dataset/chunk_size)
positive_text = []
softmax = torch.nn.Softmax(dim=1)
positive_scores = []
def inference(token_id, attention_masks, positive_text, positive_scores ,text,counter):

    d = TensorDataset(token_id, attention_masks)
    loader = DataLoader(d,batch_size = 1024)
    pred = []
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader):
            tokens = batch[0].to(device)
            att_mask = batch[1].to(device)
            sm = softmax(model(tokens, att_mask)[0])
            scores += [i[1] for i in sm.cpu().detach().numpy()]
            pred += list(np.argmax(sm.cpu().detach().numpy(), axis = 1))

    positive = list(np.where(np.array(pred)==1)[0])
    positive_text_temp = [text[i] for i in positive]
    positive_scores_temp = [scores[i] for i in positive]
    positive_scores += positive_scores_temp
    positive_text += positive_text_temp
    if counter % 1 == 0:
        ds = Dataset.from_dict({'text': positive_text, 'score':positive_scores})
        ds.save_to_disk('manufacturing_dataset_ita/checkpoint{0:05d}'.format(counter))
    print('\n\n')
    print('Run ' + str(counter) + ' completed. Found ' + str(len(positive)) + ' positive corpus\n\n')  
    
    #print(token_id[0:3])
def tokenization(text,tokens_new,attention_new):
    tokens_new.clear()
    attention_new.clear()
    t, a = tokenize_dataset(text)
    tokens_new += t
    attention_new += a


for i in range(n_it):
    if (i+1)*chunk_size<size_dataset:
        text = dataset[i*chunk_size:(i+1)*chunk_size]['text']
    else:
        text = dataset[i*chunk_size:size_dataset]['text']
    if i == 0:
        tokens_new, attention_new = tokenize_dataset(text)
    print('Inizio')
    token_id = torch.tensor(tokens_new, dtype=int)
    attention_mask = torch.tensor(attention_new, dtype=int)
    start = (i+1)*chunk_size
    end = (i+2)*chunk_size if (i+2)*chunk_size < size_dataset else size_dataset
    print("it: ",i)
    t_inference = threading.Thread(target = inference, args = (token_id, attention_mask,positive_text,positive_scores,text,i))
    if start<=size_dataset:
        t_token = threading.Thread(target = tokenization, args = (dataset[(i+1)*chunk_size:end]['text'], tokens_new, attention_new))

    t_inference.start()
    if start<=size_dataset:
        t_token.start()

    t_inference.join()
    t_token.join()

ds = Dataset.from_dict({'text': positive_text, 'score':pred})
ds.save_to_disk('manufacturing_dataset_ita/final')