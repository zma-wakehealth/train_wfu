from collections import Counter
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

        # Split BIO tag (e.g., "B-DISEASE" -> "B", "DISEASE")
        prefix, entity = tag.split("-", 1)

        if prefix == "B":
            # Start of a new entity: close previous if it exists
            if current:
                spans.append(tuple(current))
            current = [unique_id, idx, idx + 1, entity]

        elif prefix == "I":
            # Continuation: Extend if entity type matches, otherwise handle "broken" BIO
            if current and current[3] == entity:
                current[2] = idx + 1
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

    all_pred_spans = []
    all_true_spans = []

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
        
        all_pred_spans.extend(extract_spans(masked_pred_seq, int2str, unique_id))
        all_true_spans.extend(extract_spans(label_seq, int2str, unique_id))

    # --- GLOBAL (MICRO) CALCULATIONS ---
    # Counter subtraction correctly handles counts: 
    # (Preds & True) = True Positives
    # (Preds - True) = False Positives
    # (True - Preds) = False Negatives
    pred_counter = Counter(all_pred_spans)
    true_counter = Counter(all_true_spans)

    tp = sum((pred_counter & true_counter).values())
    fp = sum((pred_counter - true_counter).values())
    fn = sum((true_counter - pred_counter).values())

    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro)
                if (precision_micro + recall_micro) > 0 else 0.0)

    # --- PER-CLASS & MACRO CALCULATIONS ---
    per_class = {}
    true_class_counts = Counter([s[3] for s in all_true_spans])
    # Combine sets of labels to ensure we evaluate everything the model guessed
    labels_set = set(true_class_counts.keys()) | set([s[3] for s in all_pred_spans])

    f1_list = []
    weights = []

    for label in sorted(labels_set):
        # Filter spans for this specific class
        p_class = [s for s in all_pred_spans if s[3] == label]
        t_class = [s for s in all_true_spans if s[3] == label]

        pc = Counter(p_class)
        tc = Counter(t_class)

        tp_c = sum((pc & tc).values())
        fp_c = sum((pc - tc).values())
        fn_c = sum((tc - pc).values())

        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0

        support = true_class_counts.get(label, 0)
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
        "per_class": per_class
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
