# Project Metadata
AUTHOR_NAME="FIRST_NAME LAST_NAME"
AUTHOR_EMAIL="user@xomnia.com"

PROJECT_NAME='steam-model'
PROJECT_DESCRIPTION='Regression model to predict player counts on Steam platform'

PYTHON_VERSION = ">=3.10.10"
MODEL_VERSION='0.0.1'

# Paths
DIR_DATA = './Data'
DIR_OUTPUT = './output'
DIR_DATA_FILE = './output/df_all.csv'
DIR_MODEL_DEPLOY = './output/models'

# Columns
COLS_TARGET = ['price_mean', 'price_max',]
COLS_EXCLUDE = ['count_min', 'count_mean', 'count_max', 'discount_time_perc', 'discount_time_count', 'discount_min', 'discount_mean', 'discount_max', 'price_min', ]

REGEX_BOOL_FEATURES = 'developers|publishers|genres|tags|language'

# Parameters for Feature Engineering preprocessor
MOST_COMMON_THRESHOLD = {
    'developers': 50,
    'publishers': 50,
    'genres': 50,
    'tags': 50,
    'language': 10,
}

TIMESERIES_FILLING_METHOD = 'ffill'
TIMESERIES_AGGREGATE_GRANULARITY = 'D'

PCA_MIN_COMPONENTS = 2
PCA_RATIO_TOTAL_COLUMNS = 0.25

FEATURE_STATISTICAL_FUNCTIONS = ['min', 'mean', 'max']
FEATURE_STATISTICAL_FIELDS = [('Discount', 'discount_'), ('Finalprice', 'price_'), ('Playercount', 'count_')]

FE_DROP_CONSTANT_PARAMS={
    'tol':1,
    'missing_values':'ignore',
}

FE_SMART_CORRELATION_PARAMS={
    'threshold':0.90,
    'method':'pearson',
    'selection_method':'missing_values',
    'missing_values':'ignore',
}

FE_NUMERIC_IMPUTATION={
    'imputation_method':'median',
}

# Parameters for model training, validation and tracking
MODELS = ['PLS', 'LASSO', 'ELASTICNET', 'RANDOM_FOREST', 'MLP', 'GBM', ] # 'SVR'

TEST_SIZE=0.2
SEARCH_ITERATIONS=100

CV_SCORES=['mse', 'rmse', 'mae', 'max_error', 'r2']
CV_METRIC='mse'
CV_FOLDS=10

MLFLOW_BOOL=True
MFLOW_DIR='./mlruns/'
MLFLOW_TRACKING_URI='http://localhost:5000'
MLFLOW_EXPERIMENT_NAME='steam'

# Other Parameters
BOOL_VERBOSE=True
SEED=1
N_JOBS=-2
