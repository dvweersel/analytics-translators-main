from pandas import to_datetime

def parser(date):
    if isinstance(date, float):
        return date
    elif len(date) > 7:
        return to_datetime(date, format='%d-%b-%y')
    else: 
        return to_datetime('01-'+date, format='%d-%b-%y')