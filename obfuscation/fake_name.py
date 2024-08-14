from itertools import cycle
import numpy as np
import re
from collections import Counter

pat1 = re.compile(r'[ |,|$|\n|\(|\)]+')
credentials = ['dr', 'drs', 'ms', 'md', 'phd', 'mr', 'jr', 'sr', 'pa', 'do', 'phd', 'ii', 'iii', 'iv', 'v', '&quot;']

def get_fmt(name):
    prev_end = 0
    spans = list(pat1.finditer(name+'$'))
    is_comma_first = False
    fmt = ''
    parts = []
    for span in spans:
        curr_start, curr_end = span.span(0)
        curr_word = name[prev_end:curr_start]

        # special case
        # this is possible for name = '\n\nABC'
        if len(curr_word) == 0:
            fmt += 'S'
            parts.append(name[curr_start:curr_end])
            prev_end = curr_end
            continue

        # regular cases
        # check where the comma is
        if is_comma_first == False and curr_start < len(name) and name[curr_start] == ',':
            if fmt == '' or fmt[-1] != 'F':
                is_comma_first = True
        # for now added either C or F, then correct the F after the loop
        if curr_word.lower().replace('.','') in credentials:
            fmt += 'CS'
        else:
            fmt += 'FS'
        parts.append(curr_word)
        parts.append(name[curr_start:curr_end])
        prev_end = curr_end

    # now correct the F with L if needed
    if len(parts[-1]) == 0:
        parts = parts[:-1]
        fmt = fmt[:-1]
    counter = Counter(fmt)

    # if some errors returning only C, then just return it
    if counter['F'] == 0:
        return fmt, parts

    # if only one name, assume it's first name
    if fmt == 'F':
        return fmt, parts
    
    # if it's like Dr ABC, return Credential + Last Name
    if counter['F'] == 1 and counter['C'] >= 1:
        return fmt[:-1] + 'L', parts
    
    # if there's multiple parts, then either replace the first F with L or the last F with L
    if is_comma_first:
        for idx, c in enumerate(fmt):
            if c == 'F':
                return fmt[:idx] + 'L' + fmt[idx+1:], parts
    else:
        for idx in range(len(fmt))[::-1]:
            c = fmt[idx]
            if c == 'F':
                return fmt[:idx] + 'L' + fmt[idx+1:], parts

    return None, None  # should not get there

def parse_dot_name(fmts, parts):
    '''
      further separate name with . in it
    '''
    fmts_new, parts_new = '', []
    for fmt, part in zip(fmts, parts):
        if fmt == 'S' or fmt == 'C':
            fmts_new += fmt
            parts_new.append(part)
        else:
            splits = part.split('.')
            if len(splits) == 1:  # if no dots
                if len(part) == 1:  # also short hand the single letter name
                    fmts_new += fmt.lower()
                else:
                    fmts_new += fmt
                parts_new.append(part)
            else:
                for i, split in enumerate(splits[:-1]):
                    fmts_new += fmt.lower()
                    parts_new.append(split+'.')
    
    # deals with the apos
    if parts_new[-1][-7:] == '&apos;s':
        fmts_new += 'S'
        tmp = parts_new.pop(-1)
        parts_new.append(tmp[:-7])
        parts_new.append('&apos;s')
    elif parts_new[-1][-5:] == '&apos':
        fmts_new += 'S'
        tmp = parts_new.pop(-1)
        parts_new.append(tmp[:-5])
        parts_new.append('&apos')
    else:
        pass

    return fmts_new, parts_new

def fix_case(ref_string, output_string):
    ''' assume the output_string is camel case '''
    if ref_string[0].islower():
        return output_string.lower()
    if len(ref_string) > 1 and ref_string[1].isupper():
        return output_string.upper()
    return output_string

def print_fake_name(parts, fmts, fake_first_names, fake_last_name):
    i = 0
    output = ''
    for part, fmt in zip(parts, fmts):
        if fmt in 'C' or fmt in 'S':
            output += part
        elif fmt == 'l':
            output += fix_case(part, fake_last_name[0])
            if part[-1] == '.':  output += '.'
        elif fmt == 'L':
            output += fix_case(part, fake_last_name)
        elif fmt == 'f':
            output += fix_case(part, fake_first_names[i][0])
            i += 1
            if part[-1] == '.':  output += '.'
        else:
            output += fix_case(part, fake_first_names[i])
            i += 1
    return output


class Person():
    '''
      assume when this person inits, it has the longest list of firstnames
    '''
    def __init__(self, parts, fmts, fakename):
        self.first_names, self.last_name = [], ''
        self.fake_first_names, self.fake_last_name = [], ''
        for part, fmt in zip(parts, fmts):
            if fmt == 'S' or fmt == 'C':
                continue
            if fmt.lower() == 'f':
                self.first_names.append(part.lower())
                self.fake_first_names.append(next(fakename.fake_first_names))
            else:
                self.last_name = part.lower()
                self.fake_last_name = next(fakename.fake_last_names)
    
    def _is_short(self, part):
        if len(part) == 0:  # it could be empty string if say lastname is missing
            return False
        return len(part) == 1 or part[-1] == '.'

    def _is_same_part(self, part1, part2):
        # see if part2 matches part1
        if len(part1) == 0:
            return False
        if self._is_short(part1) or self._is_short(part2):
            return part1[0] == part2[0]
        else:
            return part1 == part2

    def is_same_person(self, parts, fmts):
        i = -1
        fake_first_names, fake_last_name = [], ''
        for part, fmt in zip(parts, fmts):
            if fmt == 'C' or fmt == 'S':
                continue
            part = part.lower()
            if fmt.lower() == 'l':
                if self._is_same_part(self.last_name, part) == False:
                    return False, None
                else:
                    fake_last_name = self.fake_last_name
            else:
                i += 1
                while i < len(self.first_names):
                    if self._is_same_part(self.first_names[i], part):
                        fake_first_names.append(self.fake_first_names[i])
                        break
                    else:
                        i += 1
                if i == len(self.first_names):
                    return False, None
        return True, print_fake_name(parts, fmts, fake_first_names, fake_last_name)

class FakeName():
    def __init__(self, filename):
        with open(filename, 'r') as fid:
            self.fake_first_names, self.fake_last_names = [], []
            for k, line in enumerate(fid):
                if k == 0:  continue
                first, last = line.split('\t')
                last = last.strip()
                self.fake_first_names.append(first)
                self.fake_last_names.append(last)
        self.fake_first_names = cycle(np.random.choice(self.fake_first_names, 10*len(self.fake_first_names), replace=True))
        self.fake_last_names = cycle(np.random.choice(self.fake_last_names, 10*len(self.fake_last_names), replace=True))
    
    def parse_and_replace(self, names):
        persons, outputs = [], []
        for name in names:
            fmts, parts = get_fmt(name)
            fmts, parts = parse_dot_name(fmts, parts)
            found = False
            for person in persons:
                # temporarily switch this to L to see if there is match
                if len(fmts) == 1:
                    tmp = person.is_same_person(parts, 'L')
                    if tmp[0]:
                        outputs.append(tmp[1])
                        found = True
                        break
                tmp = person.is_same_person(parts, fmts)
                if tmp[0]:
                    outputs.append(tmp[1])
                    found = True
                    break
            if found == False:
                person = Person(parts, fmts, self)
                outputs.append(print_fake_name(parts, fmts, person.fake_first_names, person.fake_last_name))
                persons.append(person)
        return outputs

if (__name__ == '__main__'):
    fakename = FakeName('C:/Users/zhma/Projects/deid/train_wfu/obfuscation/fake_name_list.txt')

    names = ['A H Baa', 'A. Baa', 'H Baa', 'Dr Baa', 'A.H.', 'C.A. Daa', 'C Daa', 'Ms Daa']
    print(fakename.parse_and_replace(names))

    names = ['WESLEY HSU', 'BOWLIN, MARLENE C', 'SAHUSSAPONT SIRINTRAPUN', '\n\nSharon N Sims']
    print(fakename.parse_and_replace(names))

    names = ['Quincy Alexander Cook', 'Quincy Alexander Cook', 'Quincy', 'Quincy', 'Quincy', 'COOK, QUINCY ALEXANDER', 'COOK, QUINCY ALEXANDER', 'COOK, QUINCY ALEXANDER', 'Quincy', 'Diane Samelak', 'Thomas McLean']
    print(fakename.parse_and_replace(names))

    names = ['Alexander (Alex)', 'Alex', 'Alex', 'Sascha Rhiannon Burnette', 'Sascha Rhiannon Burnette']
    print(fakename.parse_and_replace(names))