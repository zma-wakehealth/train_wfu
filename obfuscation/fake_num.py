import string
from itertools import cycle
import numpy as np

class FakeNum():
    '''
      randomly replace numbers and letters
    '''
    def __init__(self):
        self.num = set(string.digits)
        self.lower = set(string.ascii_lowercase)
        self.upper = set(string.ascii_uppercase)
        self.num_cycle = cycle(np.random.choice(list(string.digits), 10000, replace=True))
        self.lower_cycle = cycle(np.random.choice(list(string.ascii_lowercase), 10000, replace=True))
        self.upper_cycle = cycle(np.random.choice(list(string.ascii_uppercase), 10000, replace=True))
    
    def parse_and_shift(self, text):
        output = ''
        for c in text:
            if c in self.num:
                output += next(self.num_cycle)
            elif c in self.lower:
                output += next(self.lower_cycle)
            elif c in self.upper:
                output += next(self.upper_cycle)
            else:
                output += c
        return output

if (__name__ == '__main__'):
    fakenum = FakeNum()
    for _ in range(3):
      print(fakenum.parse_and_shift('# 34D0017525'))
      print(fakenum.parse_and_shift('SL19-501'))
    

                