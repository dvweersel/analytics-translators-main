import re
from unidecode import unidecode

def clean_text(txt: str, upper_case: bool = False) -> str:
    
    out = unidecode(txt)
    out = re.sub(r'[,.();:@#?!\-&$]+\ *', ' ', out).strip()  
    out = re.sub('\\s{2}',' ', out).strip()  
    out = re.sub(' ','_', out)
    
    if upper_case:
        return out.upper()
    return out.lower()