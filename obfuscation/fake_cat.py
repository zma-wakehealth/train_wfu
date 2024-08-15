from itertools import cycle
import numpy as np

class FakeCat():
    '''
      replace the LOCATION and HOSPITAL
      these are will be picked from a sample list
    '''
    def __init__(self, filename):
        with open(filename, 'r') as fid:
            lines = fid.readlines()
        lines = [x.strip() for x in lines]
        self.samples = cycle(np.random.choice(lines, 10*len(lines), replace=True))
    
    def reset(self):
        self.new_note = True
    
    def parse_and_replace(self, text):
        if self.new_note:
            self.mapping = {}
        text = text.strip()
        if text not in self.mapping:
            self.mapping[text] = next(self.samples)
        return self.mapping[text]

if (__name__ == '__main__'):
    fakecat = FakeCat(filename='./fake_hospital_list.txt')
    for _ in range(30):
        print(next(fakecat.samples))
    
    print('--- doing replacing ---')
    print(fakecat.parse_and_replace('sfsaf', new_note=True))
    print(fakecat.parse_and_replace('hellllllo'))
    print(fakecat.parse_and_replace('123'))
    print(fakecat.parse_and_replace('hellllllo'))
    print(fakecat.parse_and_replace('hellllllo  '))
    print(fakecat.parse_and_replace('hellllllo  \n'))