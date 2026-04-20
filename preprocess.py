# import numpy as np
# from sklearn.metrics import classification_report, f1_score
# from wfudata.wfudata import useful_class_names, normal_l 

class PreProcess:
    """
    Convert raw text + span annotations into BIO token-level labels
    using string formatting for label lookups.
    """

    def __init__(self, tokenizer, str2int, max_length=512, stride=64):
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > max_length:
            tokenizer.model_max_length = max_length

        self.stride = stride
        self.str2int = str2int
        # Cache 'O' to avoid formatting it repeatedly
        self.outside_label_id = self.str2int("O")

    def __call__(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=None,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            stride=self.stride,
        )

        all_labels = []

        for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
            sample_idx = tokenized_inputs["overflow_to_sample_mapping"][i]
            phis = examples["phi"][sample_idx]

            labels = []

            for start, end in offsets:
                # 1. Handle special tokens (CLS, SEP, PAD)
                # These have (0, 0) offsets in most Hugging Face tokenizers
                if start == end:
                    labels.append(-100)
                    continue

                label_id = self.outside_label_id

                # 2. Iterating through the PHI dictionary-of-lists
                # We check if the current token overlaps with any PHI span
                for s, e, t in zip(phis["start"], phis["end"], phis["type"]):
                    if start < e and end > s:
                        # 3. BIO Logic via string formatting
                        # B- if token start matches span start exactly
                        if start <= s < end:     # this is more robust since some model will attach space in token " token" 
                            label_id = self.str2int(f"B-{t}")
                        # I- if the token starts after the span start
                        else:
                            label_id = self.str2int(f"I-{t}")
                        break

                labels.append(label_id)

            all_labels.append(labels)

        tokenized_inputs["labels"] = all_labels

        # Cleanup for training
        tokenized_inputs.pop("offset_mapping")
        tokenized_inputs.pop("overflow_to_sample_mapping")

        return tokenized_inputs

# some sanity check
if (__name__ == '__main__'):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from wfudata.wfudata import LABEL2ID

    wfu_dataset = load_dataset('wfudata/wfudata.py', data_dir='./wfudata', trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained('nlpie/bio-distilbert-uncased', local_files_only=True)

    preprocess = PreProcess(tokenizer, wfu_dataset['train'].features['label'].str2int, stride=16)
    wfu_dataset_tokenized = wfu_dataset.map(preprocess, batched=True, batch_size=16, 
                                            remove_columns=wfu_dataset['train'].column_names)

    print(wfu_dataset)
    print(wfu_dataset['train'][0])

    example = wfu_dataset_tokenized['train'][0]
    tokens, int2str = [], wfu_dataset['train'].features['label'].int2str
    for input_id, label in zip(example['input_ids'], example['labels']):
        if label <= 0:  continue
        print(f"{tokenizer.decode(input_id)} {int2str(label)}")

    example = wfu_dataset_tokenized['train'][1]
    tokens, int2str = [], wfu_dataset['train'].features['label'].int2str
    for input_id, label in zip(example['input_ids'], example['labels']):
        if label <= 0:  continue
        print(f"{tokenizer.decode(input_id)} {int2str(label)}")

    # print(wfu_dataset_tokenized['train'][0])