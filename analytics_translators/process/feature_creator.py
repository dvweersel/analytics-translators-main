import datetime
import typing as t
import pandas as pd
import numpy as np

class FeatureCreator:
    
    def __init__(self, 
                 id: str, 
                 id_col: str,
                 game_data: pd.DataFrame, 
                 count_data: t.Dict[str, pd.DataFrame], 
                 price_data: t.Dict[str, pd.DataFrame],
                 date_col: str = 'releasedate',
                 name_col: str = 'name',
                 timeframe_days: int = 365,
                 *args, **kwargs,
                 ) -> None:
        
        self.id = str(id)
        self.id_col = id_col
        self.game_data = game_data
        self.count_data = count_data[self.id]
        self.price_data = price_data[self.id]
        self.date_col = date_col
        self.name_col = name_col
        self.timeframe_days = timeframe_days
        
        self.features = pd.DataFrame()
        
        self._filter_game_data()
        self._extract_metadata()
        self._merge_count_and_price_data()
        self._filter_merged_data()
        self._calculate_game_age()
        
    def _filter_game_data(self):
        """Subset game data to only relevant id"""
        self.game_data_id = self.game_data.loc[self.game_data[self.id_col] == self.id]
    
    
    def _extract_name(self):
        self.name = self.game_data_id[self.name_col]
        
    
    def _extract_release_date(self):
        self.release_date = self.game_data_id[self.date_col]
        
        
    def _extract_metadata(self):
        self._extract_name()
        self._extract_release_date()
        
            
    def _merge_count_and_price_data(self):
        self.merged_data = pd.merge(
            self.count_data,
            self.price_data,
            left_index=True,
            right_index=True,
            how='outer'
            ) \
            .assign(id = self.id)
            
                    
    def _extract_start_and_end_date(self):
        
        self.end_date = self._extract_index_date(self.merged_data, np.max)
        self.start_date = self.end_date - datetime.timedelta(days=self.timeframe_days)
        
            
    def _filter_merged_data(self):
        
        try:
            self.end_date
            self.start_date
        except AttributeError:
            self._extract_start_and_end_date()
        
        self.merged_data = self.merged_data \
            .loc[self.start_date:self.end_date] \
            .dropna()
            
                
    def _extract_index_date(self, df, func):
        
        return func(df.index)
        
        
    def _calculate_game_age(self):
        """Calculate age of game in days by the time of the starting period"""
        
        try:
            self.start_date
        except AttributeError:
            self._extract_start_and_end_date()
            
        try:
            self.release_date
        except AttributeError:
            self._extract_release_date()
            
        self.age_days = (self.start_date - self.release_date).dt.days
        self.age_days.index = [self.id]
        self.age_days.name = 'age_days'
        
    
    def create_age_feature(self):
        
        try:
            self.age_days
        except AttributeError:
            self._calculate_game_age()
            
        self.features = pd.concat([self.age_days, self.features], axis=1)
        
        
    def create_statistical_features(
        self,
        funcs: t.List[str] = ['min', 'mean', 'median', 'max'],
        fields: t.Iterable[t.Tuple[str, str]] = [('Discount', 'discount_'), ('Finalprice', 'price_'), ('Playercount', 'count_')],
        ):

        for column, prefix in fields:

            tmp = self.merged_data[column].agg(funcs).to_frame().transpose()
            tmp.columns = [prefix + stat for stat in funcs]
            tmp.index = [self.id]
            
            self.features = pd.concat([self.features, tmp], axis=1)


    def create_discount_features(
        self,
        col: str = 'Discount',
        funcs: t.List[str] = ['mean'],
        ):
        
        # Create features of time discounted
        tmp = self.merged_data[[col]] \
            .assign(discount_time_perc = self.merged_data[col] != 0) \
            .assign(discount_time_count = np.count_nonzero(np.diff(self.merged_data[col])) / 2 )  \
            .agg(funcs) \
            .drop([col], axis=1)

        tmp.index = [self.id]
        self.features = pd.concat([self.features, tmp], axis=1)
        
    
    def get_features(self):
        return self.features