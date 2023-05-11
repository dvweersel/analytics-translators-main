import pandas as pd

class CountProcessor:
    
    def __init__(self,
                 path: str,
                 params: dict = None,
                 verbose: bool = False,
                 ) -> None:
        
        
        self.path = path
        self.params = params
        self.verbose = verbose
        
    def load_dataset(self, *args, **kwargs) -> None:
        
        self._df = pd.read_csv(self.path, *args, **kwargs)
        

    def fill_missing_values(self, method: str='ffill'):
        self._df = self._df.fillna(method=method)
        
        for col in self._df.columns:        
            if self._df[col].isna().sum() > 0:
                
                if self.verbose:
                    print(f'\tBackfilling values for {col}')
                    
                self._df[col] = self._df[col].fillna(method='bfill')
            
    def aggregate_values(self, aggregate_by: str='D', *args, **kwargs):
        
        self._df_agg = self._df.copy().resample(aggregate_by, *args, **kwargs).mean().round(0)
        
        
    def get_data(self) -> pd.DataFrame:
        return self._df_agg

    @property
    def df(self):
        return self._df

    @property
    def df_agg(self):
        return self._df_agg

