# Classifiers

Repository for keeping track of the experiments and tools for building a text classifier for geenrating positive and negative examples of industrial/manifacturing datasets, towards the finetuning of a LLM.

The classifier that has been used is a fine-tuned version of BERT, called ManuBERT https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4375613. 

The file present in this repository are:

- ``run_classification.py``: script provided by HuggingFace used to fine-tune the BERT classifier on the SequenceClassification task
- ``train_bert.py``: bash script used to run the training script provided by HuggingFace 
- ``run_classifier.py``: script used to perform filter the c4 dataset provided by allenai on HuggingFace
- ``run_classifier_ita.py``: script used to perform filter the c4 dataset provided by Gabriele on HuggingFace.

The Raw dataset are availabe on the 