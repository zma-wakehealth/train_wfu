"""
Training script for WFU PHI NER (BIO tagging)

Key features:
- Hugging Face Trainer with custom loss (class weights)
- Token-level BIO labeling with span-level evaluation
- Support for cross-validation folds (wfudata_fold_X)
"""

import os
import sys
import numpy as np
import datasets

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments
)

from utils_fold import PreProcess, compute_metrics
from mytrainer import MyTrainer

# label mappings (must match dataset)
from wfudata_fold import id2label, label2id


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
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Workaround: ensure parameters are contiguous (avoids rare HF save bugs)
    for _, param in model.named_parameters():
        param.data = param.data.contiguous()

    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':

    # ---- Parse fold argument ----
    fold = sys.argv[1]

    # ---- Device ----
    device = 'cuda:0'

    # ---- Load dataset ----
    # Expect dataset script like wfudata_fold_0, wfudata_fold_1, etc.
    wfu_dataset = datasets.load_dataset(
        f'wfudata_fold_{fold}',
        trust_remote_code=True
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained('nlpie/bio-distilbert-uncased')

    # IMPORTANT:
    # Use full model context (DistilBERT supports 512 tokens)
    MAX_LENGTH = 512
    STRIDE = 128

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
    output_dir = f'with_balanced_weighting_biodistilbert_fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)

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
        per_device_train_batch_size=8,   # safer for 512 tokens
        per_device_eval_batch_size=8,

        num_train_epochs=10,
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

        fp16=True,  # use mixed precision for speed/memory

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

        compute_metrics=compute_metrics
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()
