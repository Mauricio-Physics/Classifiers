from datasets import Dataset, load_dataset, DatasetDict
import random
import math

ds_en = load_dataset('knowgen/Manufacturing_EN_cleaned') 
ds_it = load_dataset('knowgen/Manufacturing_IT_cleaned')

text_en_train = ds_en['train']['text']
text_en_test = ds_en['test']['text']
text_en_valid = ds_en['validation']['text']
text_it = ds_it['train']['text']
text = text_en_train+text_en_test+text_en_valid+text_it

idxs = [i for i in range(len(text))]
random.shuffle(idxs)

train_size = math.floor(0.8*len(text))
test_size = math.ceil(0.1*len(text))
valid_size = math.floor(0.1*len(text))
train_text = [text[idxs[i]] for i in range(train_size)]
test_text = [text[idxs[i+train_size]] for i in range(test_size)]
valid_text = [text[idxs[i+train_size+test_size]] for i in range(valid_size)]

ds_cp_train = Dataset.from_dict({'text': train_text})
ds_cp_test = Dataset.from_dict({'text': test_text})
ds_cp_valid = Dataset.from_dict({'text': valid_text})

ds_cp = DatasetDict({
    'train': ds_cp_train,
    'test': ds_cp_test,
    'validation': ds_cp_valid
    }
)

ds_cp.push_to_hub('knowgen/Continual_PreTraining')