from dateutil.parser import parse
from datetime import datetime, timedelta
from itertools import cycle

hours = cycle(['H', 'I', 'H', 'H'])  #it can't output -H on my laptop, so skip those
pms = cycle(['', ' %p'])
time_connects = cycle([':', ':', ''])

years = cycle(['y', 'Y'])
months = cycle(['m','m','m','b','B'])
date_connects = cycle(['/','/','-','/'])
date_orders = cycle(['MDY','MDY','MDY','MDY','YMD'])
B_modifiers = cycle([', ',', ',' of ',' ',' '])  # modifier for feb 13, 2020

dt_connects = cycle([' ', ' ', ' ', ' ', ' @ ', ' at '])

default1 = datetime(2024,1,1,0,0,0)
default2 = datetime(2023,12,28,23,59,59)

class FakeDate():
    '''
      parse and shift the date
    '''
    def uparse(self, text, default):
        try:
            util_parsed = parse(text, default=default)
            if util_parsed.year < 1900:
                util_parsed = datetime.strptime(text, '%H%M')
                util_parsed = util_parsed.replace(year=default.year, month=default.month, 
                                                day=default.day, second=default.second)
        except:
            util_parsed = None
        return util_parsed

    def parse_and_shift(self, date_text, shift=0):
        date_text = date_text.lower().replace('@','').replace('at','')
        util_parsed_1 = self.uparse(date_text, default1)
        util_parsed_2 = self.uparse(date_text, default2)         
    
        # if not able to parse, return the original text
        if util_parsed_1 is None or util_parsed_2 is None:
            print(f'not parsed {date_text}')
            return date_text
        
        # shift for some amount
        if shift != 0:
            delta = timedelta(seconds=shift)
            util_parsed_shifted = util_parsed_1 + delta
        else:
            util_parsed_shifted = util_parsed_1 
        
        # get the time fmt string
        hour, minute, second = '', '', ''
        if util_parsed_1.hour == util_parsed_2.hour:
            hour = '%' + next(hours)
        if util_parsed_1.minute == util_parsed_2.minute:
            minute = '%' + 'M'
        if util_parsed_1.second == util_parsed_2.second:
            second = '%' + 'S'
        if hour and hour[-1] == 'I':
            pm = ' %p'
        else:
            pm = next(pms)
        if hour == '' and minute == '' and second == '':
            fmt_time = ''
        else:
            fmt_time = next(time_connects).join([x for x in [hour, minute, second] if x != ''])
            fmt_time = fmt_time + pm
        output_time = datetime.strftime(util_parsed_shifted, fmt_time)

        # get the date fmt string
        year, month, day = '', '', ''
        date_order = next(date_orders)
        if util_parsed_1.year == util_parsed_2.year:
            year = '%' + next(years)
        if util_parsed_1.month == util_parsed_2.month:
            if date_order == 'YMD':
                month = '%m'
            else:
                month = '%' + next(months)
        if util_parsed_1.day == util_parsed_2.day:
            day = '%d'
        if day == '' and month == '' and year != '':
            year = '%Y'
        
        if month and (month[-1] == 'B' or month[-1] == 'b'):
            fmt_date = ' '.join([x for x in [month, day] if x != ''])
            if year != '':
                extra_year = next(B_modifiers) + datetime.strftime(util_parsed_shifted, year)
            else:
                extra_year = ''
            output_date = datetime.strftime(util_parsed_shifted, fmt_date) + extra_year
        else:
            if date_order == 'MDY':
                date_connect = next(date_connects)
            else:
                date_connect = '-'
            fmt_date = date_connect.join([x for x in [month, day, year] if x != ''])
            output_date = datetime.strftime(util_parsed_shifted, fmt_date)
        
        if output_time is None or output_time == '':
            return output_date
        elif output_date is None or output_date == '':
            return output_time
        else:
            return output_date + next(dt_connects) + output_time

if (__name__ == '__main__'):
    fakedate = FakeDate()

    for _ in range(3):
        print(fakedate.parse_and_shift('02/13'))

        print(fakedate.parse_and_shift('Jan 2014 @ 5:30 pm'))

        print(fakedate.parse_and_shift('2012')) # this will parse a year 2012, not 20:12

        print(fakedate.parse_and_shift('1035'))