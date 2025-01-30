from datasets import load_from_disk, Dataset
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
import random
import math
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


ds = load_from_disk('translation/checkpoint_train04100/')
text = ds['text']
idx_valid = [0,3,5,6,7,8,9,11,12,16,17,18,19,22,23,24,25,26,27,28,29,31,33,35,36,37,39,
     40,41,43,45,46,47,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,67,68,
     69,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,91,92,93,94,
     95,98,99]
text_checked = [text[i] for i in idx_valid]
ds_it = load_from_disk('manufacturing_dataset_ita/checkpoint00198')
scores_it = ds_it['score']
b = np.argwhere(np.array(scores_it)>.95)
text_it_all = ds_it['text']
text_it = [text_it_all[i[0]] for i in b]
test_cl = text_checked + text_it

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
model.to(device)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

chunk_size=1000
n_it = math.ceil(len(test_cl)/chunk_size)
embeddings = []

for i in tqdm(range(n_it)):
    if (i+1)*chunk_size<len(test_cl):
        text_local = test_cl[i*chunk_size:(i+1)*chunk_size]
    else: 
        text_local = test_cl[i*chunk_size:]
    batch_dict = tokenizer(text_local, max_length=128, padding=True, truncation=True, return_tensors='pt')
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask']), batch_size=128)

    for x,y in loader:

        #outputs = model(x.to(device),y.to(device))
        outputs = model(x.to(device), y.to(device))
        
        embeddings += average_pool(outputs.last_hidden_state.to(torch.device('cpu')), y.to(torch.device('cpu'))).to(torch.device('cpu')).detach().numpy().tolist()

    #if i%100 == 0:
    #    np.save('embeddings{0:05d}.npy'.format(i),np.array(embeddings))
        
np.save('embeddings.npy',np.array(embeddings))