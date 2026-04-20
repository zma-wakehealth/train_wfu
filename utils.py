import numpy as np

def trim_span_end(start, end, raw_text):
    """
      
    """



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