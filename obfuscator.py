from obfuscation.fake_cat import FakeCat
from obfuscation.fake_date import FakeDate
from obfuscation.fake_name import FakeName
from obfuscation.fake_num import FakeNum
import re

fake_cats = {
    'EMAIL':None,
    'HOSPITAL':None,
    'IPADDRESS':None,
    'LOCATION':None,
    'OTHER':None,
    'URL':None
}

class Obfuscator():
    def __init__(self):
        for cat in fake_cats.keys():
            fake_cats[cat] = FakeCat(f'obfuscation/fake_{cat.lower()}_list.txt')
        self.fake_cats = fake_cats
        self.fake_date = FakeDate()
        self.fake_name = FakeName('obfuscation/fake_name_list.txt')
        self.fake_num = FakeNum()
        self.age_pat = re.compile(r'\d+')
    
    def parse_and_hide(self, text, results, second_shift):
        output_text = ''
        prev_end = 0
        for _, fake_cat in self.fake_cats.items():
            fake_cat.reset()

        # first get all the names
        real_names = []
        for result in results:
            phi_type, curr_start, curr_end = result['entity_group'], result['start'], result['end']
            if phi_type == 'NAME':
                real_names.append(text[curr_start:curr_end])
        fake_names = iter(self.fake_name.parse_and_replace(real_names))
        first_cat = True
        for result in results:
            phi_type, curr_start, curr_end = result['entity_group'], result['start'], result['end']
            curr_span = text[curr_start:curr_end]

            if phi_type == 'AGE':
                try:
                    tmp_span = list(self.age_pat.finditer(curr_span))[0]
                    tmp_start, tmp_end = tmp_span.span(0)
                    age_text = curr_span[tmp_start:tmp_end]
                    age = float(age_text)
                    if age > 89:
                        hidden = curr_span[:tmp_start] + '89+' + curr_span[tmp_end:]
                    else:
                        hidden = curr_span
                except:
                    print('error in age -- {curr_span} ---')
                    hidden = curr_span
                
            
            elif phi_type == 'DATE':
                hidden = self.fake_date.parse_and_shift(curr_span, second_shift)
            
            elif phi_type == 'IDNUM' or phi_type == 'INITIALS' or phi_type == 'PHONE':
                hidden = self.fake_num.parse_and_shift(curr_span)
            
            elif phi_type == 'NAME':
                hidden = next(fake_names)
            
            else:
                hidden = self.fake_cats[phi_type].parse_and_replace(curr_span)
            
            hidden = f'<span style="color:red">{hidden}</span>'
            output_text += text[prev_end:curr_start] + hidden
            prev_end = curr_end
        
        output_text += text[prev_end:]
        return output_text
    
