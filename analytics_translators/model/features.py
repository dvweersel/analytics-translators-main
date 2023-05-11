# A feature engineering pipeline for the transformation of both numeric and non-numeric data
from analytics_translators import configs

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer, AddMissingIndicator, DropMissingData
from feature_engine.encoding import RareLabelEncoder
from feature_engine.transformation import YeoJohnsonTransformer, PowerTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, DropFeatures, DropDuplicateFeatures, RecursiveFeatureElimination

import warnings
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

def create_preprocessor(
    col_bool: list,
    col_numeric: list,
    col_power: list = ['age_days', ]
    ):

    preprocessor = Pipeline(steps=[
        
        # ----------------------
        ## Bool Features

        # # Add Missing Indicator
        # ('bool_missing_indicator', AddMissingIndicator(variables=col_bool, missing_only=True)),
        
        # Impute Missing Values
        ('bool_impute_missing', MeanMedianImputer(variables=col_bool, imputation_method='median')),
        
        # ----------------------
        ## Numerical Features
        
        # Add Missing Indicator
        ('num_missing_indicator', AddMissingIndicator(variables=col_numeric, missing_only=True)),
        
        # Impute Missing Values
        ('num_impute_missing_median', MeanMedianImputer(variables=col_numeric, imputation_method='median')),
        
        # Power Transformations
        ('num_square_age', PowerTransformer(variables=col_power, exp=2)),
        
        # Yeo-Johnson Transform
        ('num_yeojohnson', YeoJohnsonTransformer(variables=col_numeric)),
        
        # Z-score Scaling
        ('num_zscore_scaling', SklearnTransformerWrapper(variables=col_numeric, transformer=StandardScaler())),
        
        # ----------------------
        # All Features

        # Remove Highly Correlated 
        ('remove_correlated', SmartCorrelatedSelection(variables=col_numeric+col_bool, **configs.FE_SMART_CORRELATION_PARAMS)),

        # Remove Duplicates
        ('all_drop_duplicate', DropDuplicateFeatures(missing_values='ignore')),
        
        # Remove Constant
        ('all_drop_constant', DropConstantFeatures(missing_values='ignore')),
        
        # Drop Missing Data
        ('all_drop_missing', DropMissingData())
        
        ])
    
    return preprocessor
