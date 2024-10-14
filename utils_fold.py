import numpy as np
from sklearn.metrics import classification_report, f1_score
from wfudata_fold import useful_class_names, normal_l 

class PreProcess():
    '''
      It's much better if you can set the tokenizer.max_length after it's loaded
    '''
    def __init__(self, tokenizer, str2int, max_length=512, stride=64):
        self.tokenizer = tokenizer
        if tokenizer.model_max_length > max_length:
            tokenizer.model_max_length = max_length
        self.stride = stride
        self.str2int = str2int
    
    def __call__(self, examples):
        '''
          careful here, you need think of examples as a list of example within a batch
        '''

        # set the max_length to none to use tokenizer.model_max_length instead
        tokenized_inputs = self.tokenizer(examples['text'], is_split_into_words=False, truncation=True,
                                          max_length=None, return_overflowing_tokens=True,
                                          stride=self.stride)
        
        labels = []
        previous_sample_idx = None
        
        # loop over line by line in this batch
        for i in range(len(tokenized_inputs['input_ids'])):

            encoding = tokenized_inputs[i]

            # track back the corresponding phi
            current_sample_idx = tokenized_inputs['overflow_to_sample_mapping'][i]
            if type(examples) is dict:  # only one instance call
                phi = examples['phi']
            else:
                phi = examples['phi'][current_sample_idx]
            if current_sample_idx != previous_sample_idx:
                k = 0
                previous_sample_idx = current_sample_idx

            label = []
            # loop over every word in this encoding
            for j in range(len(encoding)):
                span = encoding.token_to_chars(j)  
                if span is None:
                    label.append(-100)  # the ignore_index in crossentropy
                else:
                    idx1, idx2 = span
                    # if already out of range
                    if k >= len(phi['offsets']):
                        label.append(self.str2int('NORMAL'))
                    else:
                        phi1, phi2 = phi['offsets'][k]
                        # if one of the end inside a phi span, mark it
                        # it assumes the phi bracket is always at or behind the target
                        if (idx1 >= phi1 and idx1 <= phi2) or (idx2 >= phi1 and idx2 <= phi2):
                            label.append(self.str2int(phi['types'][k]))
                        else:
                            label.append(self.str2int('NORMAL'))
                        # move to the next phi offset if needed
                        # be careful it should be <= idx2 since you already compare this location, need to move the phi now
                        while k < len(phi['offsets']) and phi['offsets'][k][1] <= idx2:
                            k += 1

            labels.append(label)
        tokenized_inputs['labels'] = labels     
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