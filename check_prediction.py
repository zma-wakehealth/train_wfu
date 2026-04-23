"""
Span-level comparison between prediction and ground truth (TEXT BASED)
======================================================================

This script:
1. Loads the raw dataset
2. Recreates tokenizer + preprocess
3. Tokenizes the TEST split using .map(preprocess)
4. Loads a trained model from disk
5. Runs inference with Trainer
6. Reports ALL span differences:
   prediction: (etype, span_text)
   ground truth: (etype, span_text)
"""

import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from preprocess import PreProcess
import datasets
from evaluation import extract_spans


LABELS_TO_LOOK_AT = {'LOCATION', 'HOSPITAL', 'ADDRESS'}

# --------------------------------------------------
# 2. Convert token span -> readable text
# --------------------------------------------------

def span_to_text(input_ids, start, end, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[start:end])

    words = []
    current = ""

    for tok in tokens:
        if tok.startswith("##"):
            current += tok[2:]
        else:
            if current:
                words.append(current)
            current = tok

    if current:
        words.append(current)

    return " ".join(words)


# --------------------------------------------------
# 3. Report ALL span differences (no FP/FN split)
# --------------------------------------------------

def report_span_differences(
    predictions,
    labels,
    input_ids,
    int2str,
    tokenizer,
    max_samples=20,
):
    pred_ids = np.argmax(predictions, axis=-1)
    shown = 0

    for i, (p_seq, l_seq, ids_seq) in enumerate(
        zip(pred_ids, labels, input_ids)
    ):
        # mask preds where labels are masked
        masked_preds = [
            p if l != -100 else -100
            for p, l in zip(p_seq, l_seq)
        ]

        pred_spans = extract_spans(masked_preds, int2str, i)
        gold_spans = extract_spans(l_seq, int2str, i)

        pred_map = {(s, e): ent for (_, s, e, ent) in pred_spans}
        gold_map = {(s, e): ent for (_, s, e, ent) in gold_spans}

        all_locs = set(pred_map) | set(gold_map)

        printed = False

        for (start, end) in sorted(all_locs):
            p_ent = pred_map.get((start, end))
            g_ent = gold_map.get((start, end))

            # skip ent not interested it
            if p_ent not in LABELS_TO_LOOK_AT and g_ent not in LABELS_TO_LOOK_AT:
                continue

            if p_ent == g_ent:
                continue

            if not printed:
                print(f"\n================ Sample {i} ================")
                printed = True

            text = span_to_text(ids_seq, start, end, tokenizer)

            pred_str = f"({p_ent}, '{text}')" if p_ent else "None"
            gold_str = f"({g_ent}, '{text}')" if g_ent else "None"

            print(f"prediction: {pred_str} | ground truth: {gold_str}")

        if printed:
            shown += 1
            if shown >= max_samples:
                print(f"\nStopped after {max_samples} samples.")
                return


# --------------------------------------------------
# 4. Tokenizer + preprocess (IDENTICAL to training)
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "nlpie/bio-distilbert-uncased",
    local_files_only=True
)

MAX_LENGTH = 128
STRIDE = 32
tokenizer.model_max_length = MAX_LENGTH

wfu_dataset = datasets.load_dataset(
    './wfudata/wfudata.py',
    data_dir='./wfudata/data',
    trust_remote_code=True
)

preprocess = PreProcess(
    tokenizer,
    wfu_dataset["train"].features["label"].str2int,
    max_length=MAX_LENGTH,
    stride=STRIDE,
)

# --------------------------------------------------
# 5. Tokenize TEST set
# --------------------------------------------------

wfu_dataset_tokenized = wfu_dataset.map(
    preprocess,
    batched=True,
    batch_size=16,
    remove_columns=wfu_dataset["train"].column_names,
)

test_dataset = wfu_dataset_tokenized["test"]

data_collator = DataCollatorForTokenClassification(tokenizer)

# --------------------------------------------------
# 6. Load trained model
# --------------------------------------------------

output_dir = "debug/checkpoint-31140"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForTokenClassification.from_pretrained(
    output_dir
).to(device)

# --------------------------------------------------
# 7. Trainer (inference only)
# --------------------------------------------------

infer_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=16,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=infer_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --------------------------------------------------
# 8. Run inference
# --------------------------------------------------

pred_output = trainer.predict(test_dataset)

int2str = wfu_dataset["train"].features["label"].int2str
num_labels = len(wfu_dataset['train'].features['label'].names)
if not isinstance(int2str, dict):
    int2str = {i: int2str([i])[0] for i in range(num_labels)}

# --------------------------------------------------
# 9. Report span differences WITH TEXT
# --------------------------------------------------

report_span_differences(
    predictions=pred_output.predictions,
    labels=pred_output.label_ids,
    input_ids=test_dataset["input_ids"],
    int2str=int2str,
    tokenizer=tokenizer,
    max_samples=200,
)