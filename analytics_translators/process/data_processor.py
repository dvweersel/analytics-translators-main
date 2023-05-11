import pandas as pd
import numpy as np
import typing as t
import re

from collections import Counter
from unidecode import unidecode

from analytics_translators.utils.text import clean_text

class DataProcessor:
    
    def __init__(self,
                 path: str,
                 include: list,
                 params: dict = None,
                 ) -> None:
        
        
        self.path = path
        self.include = include
        self.params = params
            

    def load_dataset(self, 
                     line_pattern: str=' [,\.]|[,\.] |\\s{2,}', 
                     line_repl: str=' ', 
                     regex_remove: str = '"|\.|\'',
                     regex_repl: str = '\(MAC\)|\(LINUX\)|\(WINDOWS\)|, INC|, LLC|, ACE|, LTD|, GMBH| SRL| SRO',
                     *args, **kwargs) -> None:        

        self.entities_dict = {}
        self.entities_count = Counter()

        with open(self.path, 'r', encoding='latin-1') as f:
            
            for l in f.readlines():      
                
                l = l.upper().strip('\n')
                l = re.sub(string=l, pattern=regex_remove, repl='')
                l = re.sub(string=l, pattern=regex_repl, repl=' ')

                # Add missing comma to entries without values
                if re.search(string=l, pattern=',') is None:
                    l = l+','
                
                l = re.sub(string=l, pattern=line_pattern, repl=line_repl)
                
                tmp = l.split(',')
                out = [clean_text(x, upper_case=True) if x != '' else None for x in tmp]
                
                if (out[0] in self.include) and (out[1] is not None):        
                    entities = list(set(out[1:]))
                    entities.sort()
                    
                    self.entities_dict[out[0]] = entities
                    self.entities_count.update(out[1:])
        
        
    def get_entities_list(self) -> None:
        self.all_entities = list(self.entities_count.keys())
        self.all_entities.sort()
        
        
    def save_as_dataframe(self) -> None:
        
        try:
            self.entities_dict
        except NameError:
            self.load_dataset()
        
        try:
            self.all_entities
        except NameError:
            self.get_entities_list()
        
        entities_dict = {}

        self.df_entities = pd.DataFrame(index=self.all_entities)

        for id, entities in self.entities_dict.items():
                
            app_data = pd.Series(np.isin(self.all_entities, entities), name=id, index=self.all_entities)    
            self.df_entities = pd.concat([self.df_entities, app_data], axis=1)
            
            entities_dict[id] = app_data

        self.df_entities = self.df_entities.transpose()
        
        return self.df_entities
                
                