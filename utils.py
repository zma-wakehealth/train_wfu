<<<<<<< HEAD
class PreProcess():
    def __init__(self, tokenizer, str2int, max_length=128, stride=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.str2int = str2int
    
    def __call__(self, examples):
        '''
          careful here, you need think of examples as a list of example within a batch
        '''
        tokenized_inputs = self.tokenizer(examples['text'], is_split_into_words=False, truncation=True,
                                          max_length=self.max_length, return_overflowing_tokens=True,
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
=======
class PreProcess():
    def __init__(self, tokenizer, str2int, max_length=128, stride=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.str2int = str2int
    
    def __call__(self, examples):
        '''
          careful here, you need think of examples as a list of example within a batch
        '''
        tokenized_inputs = self.tokenizer(examples['text'], is_split_into_words=False, truncation=True,
                                          max_length=self.max_length, return_overflowing_tokens=True,
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
>>>>>>> 9666ed0422ced76cca2062f97a4a10a6538e7a75
