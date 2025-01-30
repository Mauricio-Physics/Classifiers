from datasets import load_from_disk, Dataset
from datasets.dataset_dict import DatasetDict
from matplotlib import pyplot as plt
import numpy as np
import random
import math

#---------------------------------------------------------------------------------
#Load positive documents dataset
#---------------------------------------------------------------------------------
print('Loading positve documents dataset...')
ds = load_from_disk('manufacturing_dataset_ita/positive_en_text_with_scores/final')
scores = ds['score']
text = ds['text']
idx = np.argwhere(np.array(scores)>.9)
positive_text = [text[i[0]] for i in idx]
positive_labels = list(np.ones(len(positive_text), dtype=int))

#---------------------------------------------------------------------------------
#Load negative documents dataset
#---------------------------------------------------------------------------------
print('Loading negative documents dataset...')
ds_negative = load_from_disk('manufacturing_dataset/checkpoint_false00006')
negative_text = ds_negative['text']
negative_labels = list(np.zeros(len(negative_text), dtype=int))

#---------------------------------------------------------------------------------
#Splitting dataset (train, test and validation)
#---------------------------------------------------------------------------------
print('Splitting dataset...')
ds_text = positive_text+negative_text
ds_label = positive_labels+negative_labels
idx_all = [i for i in range(len(ds_text))]
random.shuffle(idx_all)
len_train = math.ceil(len(idx_all)*0.8)
len_validation = math.floor((len(idx_all)-len_train)/2)
len_test = math.ceil((len(idx_all)-len_train)/2)

train_idx = idx_all[:len_train]
train_text = [ds_text[i] for i in train_idx]
train_labels = [ds_label[i] for i in train_idx]

test_idx = idx_all[len_train:len_train+len_test]
test_text = [ds_text[i] for i in test_idx]
test_labels = [ds_label[i] for i in test_idx]

validation_idx = idx_all[len_train+len_test:]
validation_text = [ds_text[i] for i in validation_idx]
validation_labels = [ds_label[i] for i in validation_idx]


#---------------------------------------------------------------------------------
#Creating huggingface dataset and push to hub
#---------------------------------------------------------------------------------
print('Creating HuggingFace dataset...')
ds_train = Dataset.from_dict({'text': train_text, 'label': train_labels})
ds_test = Dataset.from_dict({'text': test_text, 'label': test_labels})
ds_validation = Dataset.from_dict({'text': validation_text, 'label': validation_labels})

print('Pushing to hub...')
ds_finetuning = DatasetDict({
    'train': ds_train,
    'test': ds_test,
    'validation': ds_validation
})

ds_finetuning.push_to_hub('knowgen/mDeBERTaDataset')