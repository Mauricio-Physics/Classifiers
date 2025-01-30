export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_classification.py \
    --model_name_or_path  akumar33/ManuBERT \
    --dataset_name knowgen/ManuBERT-FineTuning \
    --text_column_name "text" \
    --do_train \
    --do_eval \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --max_seq_length 128 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --pad_to_max_length \
    --output_dir ManuBERT \
