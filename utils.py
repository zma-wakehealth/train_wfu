import numpy as np
import html

def trim_span_end(start, end, raw_text):
    """
    If a span accidentally caught the first part of an HTML entity 
    (like '&apos') because the annotator wanted to exclude the "'s", 
    this shrinks the span backward to drop the HTML junk.
    """
    # 1. Get the text currently inside the span
    span_text = raw_text[start:end]
    
    # List of common clinical entities that might be trailing
    fragments = ('&apos', '&amp', '&quot', '&lt', '&gt', '&nbsp')
    
    # 2. Check if the span ends with any of these fragments
    for frag in fragments:
        if span_text.endswith(frag):
            # Shrink the end index by the length of the fragment
            return end - len(frag)
            
    # Optional: What if they accidentally highlighted the whole '&apos;'?
    # e.g., the raw span is "Austin&apos;" but you still just want "Austin"
    full_fragments = [f + ';' for f in fragments]
    for frag in full_fragments:
        if span_text.endswith(frag):
            return end - len(frag)
            
    return end

def get_mapping_offset(original_idx, raw_text, replace_newlines=True):
    """
    Maps a raw index to the cleaned/unescaped index.
    """
    # 1. Get the prefix up to the index
    prefix = raw_text[:original_idx]
    
    # 2. Unescape (Since raw_text is healthy, this works perfectly)
    clean_prefix = html.unescape(prefix)
    
    # 3. Standardize whitespace
    clean_prefix = clean_prefix.replace('\r', ' ').replace('\t', ' ')
    
    if replace_newlines:
        clean_prefix = clean_prefix.replace('\n', ' ')
        
    return len(clean_prefix)


def compute_class_weight_ner(labels, num_classes, o_label_id=0, o_weight_scale=0.5):
    '''
      compute weight class with extra downweight on O label
    '''
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    counts = np.where(counts==0, 1, counts)
    weights = total / (num_classes * counts)

    weights[o_label_id] *= o_weight_scale

    return weights
