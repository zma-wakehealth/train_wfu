from datetime import datetime, timedelta

# see https://stackoverflow.com/questions/41191365/python-datetime-strptime-error-is-a-bad-directive-in-format-m-d-y-h
# and https://www.geeksforgeeks.org/python-datetime-strptime-function/
# the - is not needed

date_format_list = [
    '%m/%d',
    '%m/%d/%y',
    '%m/%d/%Y',
    '%m/%Y',
    '%Y',
    '%B',
    '%b',
    '%B %d',
    '%b %d',
    '%B of %Y',
    '%B %Y',
    '%b %Y',
    '%B, %Y',
    '%b, %Y',
    '%b %d, %Y',
    '%B %d, %Y',
    '%b %d %Y',
    '%B %d %Y',
    '%m-%d-%y',
    '%y-%m-%d',
    '%Y-%m-%d'
]

time_format_list = [
    '%H%M',
    '%H:%M',
    '%I:%M %p',
    '%I:%M:%S %p',   # this needs to before %H %p
    '%-I:%M %p',
    '%H:%M %p',
    '%H:%M:%S',
    '%H:%M:%S %p',
    '%H %p'
]

class FakeDate():
    # def sure_date(self, string):
    #     if string.find('/') >= 0 or string.find('-') >= 0:
    #         return True
    #     else:
    #         return False

    def sure_time(self, string):
        if string.find(':') >= 0:
            return True
        else:
            return False

    def parse_date(self, string):
        for fmt in date_format_list:
            try:
                return datetime.strptime(string, fmt), fmt
            except:
                pass
        return None, None

    def parse_time(self, string):
        for fmt in time_format_list:
            try:
                return datetime.strptime(string, fmt), fmt
            except:
                pass
        return None, None

    def parse(self, string):
        string = string.lower().strip().replace('@',' ').replace('at', ' ').replace('.', ' ')
        splits = string.split()

        # there's ambuguity in parsing 2020
        # try dates first
        if len(splits) == 1:
            dt_date, fmt_date = self.parse_date(string)
            if dt_date is not None:
                dt_time, fmt_time = None, None
            else:
                dt_time, fmt_time = self.parse_time(string)
        else:
            # first checks if there's a time stamp
            if string[-2:].lower() == 'am' or string[-2:].lower() == 'pm':
                dt_time, fmt_time = self.parse_time(' '.join(splits[-2:]))
                dt_date, fmt_date = self.parse_date(' '.join(splits[:-2]))
            elif self.sure_time(splits[-1]):
                dt_time, fmt_time = self.parse_time(splits[-1])
                dt_date, fmt_date = self.parse_date(' '.join(splits[:-1]))
            else:
                dt_date, fmt_date = self.parse_date(string)
                dt_time, fmt_time = None, None
        
            # deal with things like date 1510
            if dt_date is None and dt_time is None:
                dt_time, fmt_time = self.parse_time(splits[-1])
                dt_date, fmt_date = self.parse_date(' '.join(splits[:-1]))

        return dt_date, dt_time, fmt_date, fmt_time
    
    def shift(self, dt_date, dt_time, fmt_date, fmt_time, delta_days, delta_seconds):
        if dt_date is not None:
            str_date = datetime.strftime(dt_date + timedelta(days=delta_days), fmt_date)
        else:
            str_date = ''
        if dt_time is not None:
            str_time = datetime.strftime(dt_time + timedelta(seconds=delta_seconds), fmt_time)
        else:
            str_time = ''
        return ' '.join([str_date, str_time])


if (__name__ == '__main__'):
    fakedate = FakeDate()
    dt_date, dt_time, fmt_date, fmt_time = fakedate.parse('October 2020')
    print(fakedate.shift(dt_date, dt_time, fmt_date, fmt_time, -1, 100))


        
    