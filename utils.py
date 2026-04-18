import numpy as np
from sklearn.metrics import classification_report, f1_score
from wfudata.wfudata import useful_class_names, normal_l 

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

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        p for p, l in zip(predictions.reshape(-1), labels.reshape(-1)) if l != -100 and l != normal_l
    ]

    true_labels = [
        l for p, l in zip(predictions.reshape(-1), labels.reshape(-1)) if l != -100 and l != normal_l
    ]

    report = classification_report(true_labels, true_predictions, zero_division=0.0,
                                target_names=useful_class_names, digits=3, labels=range(len(useful_class_names)))

    return {
        "f1_macro": f1_score(true_labels, true_predictions, average='macro', labels=range(len(useful_class_names))),
        "f1_weighted": f1_score(true_labels, true_predictions, average='weighted', labels=range(len(useful_class_names))),
        "report": report
    }
