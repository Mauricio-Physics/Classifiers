import datasets 

import torch
from transformers import pipeline
from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np

#---------------------------------------------------------------------------------------------------
#Dataset load
#---------------------------------------------------------------------------------------------------
ds_train = load_from_disk('manufacturing_dataset_ita/positive_en_text_with_scores/final')
idx = np.argwhere(np.array(ds_train['score'])>np.quantile(ds_train['score'],0.99))
text = ds_train['text']
selected_text = [text[i[0]] for i in idx]
print(len(idx))
ds_train = Dataset.from_dict({'text':selected_text})

#---------------------------------------------------------------------------------------------------
#Model definition
#---------------------------------------------------------------------------------------------------
from transformers.pipelines.pt_utils import KeyDataset
#pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.float16, device_map="auto",batch_size=4)
pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.float16, device_map="auto",batch_size=4,return_full_text=False)
def templating(record):
    t = word_tokenize(record['text'])
    record['text'] = ' '.join(t[0:320] if len(t)>=320 else t)
    messages = [
                {"role": "user", "content": "Traduci dall'inglese all'italiano .\nInglese:"+record['text']+"\nItaliano:"},
    ]
    record['text'] = pipe.tokenizer.apply_chat_template(messages , tokenize=False, add_generation_prompt=True, )
    return record 

#---------------------------------------------------------------------------------------------------
#Translation traning
#---------------------------------------------------------------------------------------------------
ds_train = ds_train.map(templating, num_proc=32)
final_train_text = []


print(ds_train['text'][0])
print('\n\n')
print(ds_train['text'][10])
print('\n\n')
print(ds_train['text'][100])

print(ds_train)

i = 0 
for out in tqdm(pipe(KeyDataset(ds_train, "text"),max_new_tokens=512, do_sample=False, pad_token_id= 32005)):
    final_train_text+=[out[0]['generated_text']]
    i+=1
    if i%5 == 0 or i==1:
        print("Salvataggio it ", i) 
        ds_to_save = Dataset.from_dict({'text': final_train_text})
        ds_to_save.save_to_disk('translation/checkpoint_train{0:05d}'.format(i))
    if i == 10000:
        break


ds_to_save = Dataset.from_dict({'text': final_train_text})
ds_to_save.save_to_disk('translation/final_train')

print(ds_to_save)

#---------------------------------------------------------------------------------------------------
#Translation traning
#---------------------------------------------------------------------------------------------------
ds_test = ds_test.map(templating, num_proc=32)
final_test_text = []
final_test_label = ds_test['label']

print(ds_test)
print('\n\n')
i = 0 
for out in tqdm(pipe(KeyDataset(ds_test, "text"),max_new_tokens=256, do_sample=False, pad_token_id=32005)):
    final_test_text+=[out[0]['generated_text']]
    i+=1
    if i%5000 == 0:
        print("Salvataggio it ", i) 
        ds_to_save = Dataset.from_dict({'text': final_test_text, 'label':final_test_label[:i]})
        ds_to_save.save_to_disk('translation/checkpoint_test{0:05d}'.format(i))


ds_to_save = Dataset.from_dict({'text': final_test_text, 'label':final_test_label})
ds_to_save.save_to_disk('translation/final_test')

print(ds_to_save)


#---------------------------------------------------------------------------------------------------
#Translation traning
#---------------------------------------------------------------------------------------------------
ds_valid = ds_valid.map(templating, num_proc=32)
final_valid_text = []
final_valid_label = ds_valid['label']

print(ds_valid)
print('\n\n')
i = 0 
for out in tqdm(pipe(KeyDataset(ds_valid, "text"),max_new_tokens=256, do_sample=False, pad_token_id=32005)):
    final_valid_text+=[out[0]['generated_text']]
    i+=1
    if i%5000 == 0:
        print("Salvataggio it ", i) 
        ds_to_save = Dataset.from_dict({'text': final_valid_text, 'label':final_valid_label[:i]})
        ds_to_save.save_to_disk('translation/checkpoint_valid{0:05d}'.format(i))


ds_to_save = Dataset.from_dict({'text': final_valid_text, 'label':final_valid_label})
ds_to_save.save_to_disk('translation/final_valid')

print(ds_to_save)

#---------------------------------------------------------------------------------------------------
#Saving model
#---------------------------------------------------------------------------------------------------

data = DatasetDict({
    'train': ds_train,
    'test': ds_test,
    'valid': ds_valid})
data.save_to_disk('translation/final_dataset')