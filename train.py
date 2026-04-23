"""
Training script for WFU PHI NER (BIO tagging)

Key features:
- Hugging Face Trainer with custom loss (class weights)
- Token-level BIO labeling with span-level evaluation
"""

import os
import shutil
import numpy as np
import datasets

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments
)

from preprocess import PreProcess
from evaluation import compute_metrics
from mytrainer import MyTrainer

# label mappings (must match dataset)
from wfudata.wfudata import ID2LABEL, LABEL2ID


# -----------------------------
# Model initialization
# -----------------------------
def reset_model(num_labels, device):
    """
    Load pretrained model and reset classification head.

    Args:
        num_labels: number of BIO labels
        device: cuda or cpu

    Returns:
        model on target device
    """
    model = AutoModelForTokenClassification.from_pretrained(
        'nlpie/bio-distilbert-uncased',
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True
    ).to(device)

    # Workaround: ensure parameters are contiguous (avoids rare HF save bugs)
    for _, param in model.named_parameters():
        param.data = param.data.contiguous()

    return model


# helper function to get the evaluation
def build_compute_metrics(int2str, num_labels):
    # normalize int2str into dict for easier handling
    if not isinstance(int2str, dict):
        int2str = {i: int2str([i])[0] for i in range(num_labels)}

    def compute_metrics_with_mapping(p):
        result = compute_metrics(p, int2str)
        return result
    
    return compute_metrics_with_mapping

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':

    # ---- Device ----
    device = 'cuda:0'

    # ---- Load dataset ----
    wfu_dataset = datasets.load_dataset(
        './wfudata/wfudata.py',
        data_dir='./wfudata/data',
        trust_remote_code=True
    )

    # --- For debugging, we can use smaller samples ----
    # wfu_dataset = datasets.DatasetDict({
    #     'train': wfu_dataset['train'].select(range(100)),
    #     'test': wfu_dataset['test'].select(range(100))
    # })

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained('nlpie/bio-distilbert-uncased', local_files_only=True)

    # force the tokenizer model length to be biomert, this is from model.config['max_position_embeddings']
    # the default tokenizer.model_max_length is way too large
    # 128 reduce the memory issue
    MAX_LENGTH = 128
    tokenizer.model_max_length = MAX_LENGTH
    STRIDE = 32

    # ---- Preprocessing ----
    preprocess = PreProcess(
        tokenizer,
        wfu_dataset['train'].features['label'].str2int,
        max_length=MAX_LENGTH,
        stride=STRIDE
    )

    # Tokenize dataset with sliding window
    wfu_dataset_tokenized = wfu_dataset.map(
        preprocess,
        batched=True,
        batch_size=16,
        remove_columns=wfu_dataset['train'].column_names
    )

    # ---- Data collator ----
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # ---- Label info ----
    class_labels = wfu_dataset['train'].features['label'].names
    num_labels = len(class_labels)

    # -----------------------------
    # Class weights (balanced)
    # -----------------------------
    # Collect all valid labels (exclude -100)
    tokens_stat = []
    for seq in wfu_dataset_tokenized['train']['labels']:
        tokens_stat.extend([i for i in seq if i != -100])

    # Compute balanced weights manually (no sklearn dependency)
    counts = np.bincount(tokens_stat, minlength=num_labels)
    total = counts.sum()
    counts = np.where(counts == 0, 1, counts)

    class_weights = total / (num_labels * counts)
    class_weights = class_weights.astype(np.float32)

    print("Class weights:")
    print(class_weights)

    # -----------------------------
    # Output + logging
    # -----------------------------
    output_dir = f'debug/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    log_file = f'{output_dir}/log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)

    # -----------------------------
    # Model
    # -----------------------------
    model = reset_model(num_labels, device)

    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=30,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=50,
        log_level='info',

        save_only_model=True,
        save_total_limit=5,

        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",  # IMPORTANT: must match HF naming
        greater_is_better=True,

        push_to_hub=False
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = MyTrainer(
        class_weights=class_weights,
        log_file=log_file,

        model=model,
        args=training_args,

        train_dataset=wfu_dataset_tokenized["train"],
        eval_dataset=wfu_dataset_tokenized["test"],

        tokenizer=tokenizer,
        data_collator=data_collator,

        compute_metrics=build_compute_metrics(
            wfu_dataset['train'].features['label'].int2str,
            num_labels
        )
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()
