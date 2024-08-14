from itertools import cycle
import numpy as np

class FakeCat():
    '''
      replace the LOCATION and HOSPITAL
      these are will be picked from a sample list
    '''
    def __init__(self, pre_computed=True, filename=None, texts=None):
        if pre_computed:
            with open(filename, 'r') as fid:
                lines = fid.read()
            lines = lines.split('---')
            self.samples = cycle(np.random.choice(lines, 10*len(lines), replace=True))
        else:
            self.samples = cycle(np.random.choice(texts, 10*len(texts), replace=True))
            with open(filename, 'w') as fid:
                fid.write('---'.join(texts))
    
    def parse_and_replace(self, text, new_note=False):
        if new_note == True:
            self.mapping = {}
        text = text.strip()
        if text not in self.mapping:
            self.mapping[text] = next(self.samples)
        return self.mapping[text]

if (__name__ == '__main__'):
    texts = ['apple','###org','ndna','dum\n\n','tmp\nafda', 'i am here hello you']
    fakecat = FakeCat(pre_computed=False, filename='tmp.txt', texts=texts)

    fakecat = FakeCat(pre_computed=True, filename='tmp.txt')
    for _ in range(30):
        print(next(fakecat.samples))
    
    print('doing replacing')
    print(fakecat.parse_and_replace('sfsaf', new_note=True))
    print(fakecat.parse_and_replace('hellllllo'))
    print(fakecat.parse_and_replace('123'))
    print(fakecat.parse_and_replace('hellllllo'))
    print(fakecat.parse_and_replace('hellllllo  '))
    print(fakecat.parse_and_replace('hellllllo  \n'))