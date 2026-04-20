import numpy as np

def extract_spans(label_ids, int2str, unique_id=0):
    """
    Extracts contiguous entity spans from a sequence using BIO formatting.
    
    Args:
        label_ids: List of integer labels (including -100 for masked tokens).
        int2str: Dict or callable to map ID -> Tag (e.g., 1 -> "B-PER").
        unique_id: A unique identifier for the sequence/window to prevent 
                   collisions when aggregating results in Counter.
    
    Returns:
        List of tuples: (unique_id, start_idx, end_idx, entity_type)
        The index follows standard convention, [start_idx, end_idx)
    """
    spans = []
    current = None  # Format: [unique_id, start, end, entity_type]

    for idx, label_id in enumerate(label_ids):
        # Skip special tokens, padding, or subwords masked with -100
        if label_id == -100:
            continue

        tag = int2str[label_id] if isinstance(int2str, dict) else int2str(label_id)

        # "O" tag: Outside any entity; close the current span if one is open
        if tag == "O":
            if current:
                spans.append(tuple(current))
                current = None
            continue

        # Split BIO tag (e.g., "B-NAME" -> "B", "NAME")
        prefix, entity = tag.split("-", 1)

        if prefix == "B":
            # Start of a new entity: close previous if it exists
            if current:
                spans.append(tuple(current))
            current = [unique_id, idx, idx + 1, entity]

        elif prefix == "I":
            # Continuation: Extend if entity type matches, otherwise handle "broken" BIO
            if current and current[3] == entity:
                current[2] = idx + 1   # note that if there's a -100 in the middle of the span, this correctly continues to expand current
            else:
                # Logic for "I" without a valid "B" or type mismatch:
                # Close previous and treat this "I" as a new start (lenient parsing)
                if current:
                    spans.append(tuple(current))
                current = [unique_id, idx, idx + 1, entity]

    # Catch the final span if the sequence ends while an entity is open
    if current:
        spans.append(tuple(current))

    return spans


def compute_metrics(p, int2str):
    """
    Computes exact-match span metrics (Precision, Recall, F1).
    Handles micro, macro, and weighted averages.
    """
    predictions, labels = p
    # Convert logits to class indices
    predictions = np.argmax(predictions, axis=2)

    # I believe there's no duplicated entry, so set would be fine
    all_pred_spans = set()
    all_true_spans = set()

    # Iterate through batch
    for i, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
        # PREVENT COLLISIONS: Using loop index as a unique_id.
        # This ensures that a span (0, 5, "ORG") in Sentence A is distinct 
        # from (0, 5, "ORG") in Sentence B.
        unique_id = i 

        # Mask predictions where labels are -100 (padding/subwords)
        masked_pred_seq = [
            p_val if l_val != -100 else -100 
            for p_val, l_val in zip(pred_seq, label_seq)
        ]
        
        all_pred_spans.update(extract_spans(masked_pred_seq, int2str, unique_id))
        all_true_spans.update(extract_spans(label_seq, int2str, unique_id))

    # --- GLOBAL (MICRO) CALCULATIONS ---
    # Set operation should correctly handle: 
    # (Preds & True) = True Positives
    # (Preds - True) = False Positives
    # (True - Preds) = False Negatives
    tp = len(all_pred_spans & all_true_spans)
    fp = len(all_pred_spans - all_true_spans)
    fn = len(all_true_spans - all_pred_spans)

    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if (precision_micro + recall_micro) > 0 else 0.0)

    # --- PER-CLASS & MACRO CALCULATIONS ---
    per_class = {}
    labels_set = {s[3] for s in all_pred_spans} | {s[3] for s in all_true_spans}

    f1_list = []
    weights = []

    for label in sorted(labels_set):
        # Filter spans for this specific class
        p_class = {s for s in all_pred_spans if s[3] == label}
        t_class = {s for s in all_true_spans if s[3] == label}

        tp_c = len(p_class & t_class)
        fp_c = len(p_class - t_class)
        fn_c = len(t_class - p_class)

        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0

        support = len(t_class)
        per_class[label] = {"precision": p_c, "recall": r_c, "f1": f_c, "support": support}

        f1_list.append(f_c)
        weights.append(support)

    f1_macro = np.mean(f1_list) if f1_list else 0.0
    f1_weighted = np.average(f1_list, weights=weights) if sum(weights) > 0 else 0.0

    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        'report': format_report(per_class)
    }


def format_report(per_class):
    """
    Takes the per_class dictionary output from compute_metrics 
    and formats it into a readable string report.
    """
    lines = []
    for label, metrics in per_class.items():
        lines.append(
            f"{label:15} "
            f"P: {metrics['precision']:.3f} "
            f"R: {metrics['recall']:.3f} "
            f"F1: {metrics['f1']:.3f} "
            f"n: {metrics['support']}"
        )
    return "\n".join(lines)

# some tests here for now
if (__name__ == '__main__'):
    import torch

    # include name and date only    
    int2str = {
        0: "O",
        1: "B-NAME",
        2: "I-NAME",
        3: "B-DATE",
        4: "I-DATE",
    }

    text_1 = 'John Doe was born 1990 .'
    
    labels_1 = [
        1,  # B-NAME
        2,  # I-NAME
        0,  # O
        0,  # O
        3,  # B-DATE
        0,  # O
    ]

    preds_1 = [
        [0.0, 1.0, 0.0, 0.0, 0.0],  # B-NAME
        [0.0, 0.0, 1.0, 0.0, 0.0],  # I-NAME
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O
    ]

    text_2 = 'On   Jan   2020   .'

    
    labels_2 = [
        0,  # O
        3,  # B-DATE
        4,  # I-DATE
        0,  # O
        -100,  # padding
        -100
    ]

    
    preds_2 = [
        [0.0, 1.0, 0.0, 0.0, 0.0],  # B-NAME  (false positive)
        [0.0, 0.0, 0.0, 1.0, 0.0],  # B-DATE
        [0.0, 0.0, 0.0, 0.0, 1.0],  # I-DATE
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O should not matter
        [1.0, 0.0, 0.0, 0.0, 0.0],  # O should not matter
    ]

    predictions = np.array([
        preds_1,
        preds_2,
    ])

    labels = np.array([
        labels_1,
        labels_2,
    ])

    eval_result = compute_metrics((predictions, labels), int2str)
    print(eval_result)

    print('expected micro precision=0.667, recall=0.667, f1=0.667')
    print('expected f1 macro=0.667, f1 weighted avg=0.667')
    print('expected DATE precision=1.0, recall=0.5, f1=0.667')
    print('expected name precision=0.5, recall=1.0, f1=0.667')

