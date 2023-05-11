# Contains model object and the parameters to be used for hyperparameter tuning

import numpy as np

from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from analytics_translators import configs

model_objects = {
    'PLS': {
        'model': PLSRegression(
            max_iter=1000
            ),
        'params': {
            'model__regressor__n_components': range(1,11),
        }
    },
    'LASSO': {
        'model': Lasso(
            random_state=configs.SEED,
        ),
        'params': {
            'model__regressor__max_iter': [500, 1000],
            'model__regressor__alpha': np.logspace(-6, 6, 13),
        }
    },
    'ELASTICNET': {
        'model': ElasticNet(
            random_state=configs.SEED,
            max_iter=1000,
        ),
        'params': {
            'model__regressor__alpha': np.logspace(-6, 6, 13),
            'model__regressor__l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9,],
        }
    },
    'SVR': {
        'model': SVR(
            verbose=False
        ),
        'params': {
            'model__regressor__kernel': ['linear', 'poly', 'rbf'],
            'model__regressor__degree': [2,3,4],
            'model__regressor__C': np.logspace(-3, 3, 7),
            'model__regressor__tol': np.logspace(-3, -1, 3),
            'model__regressor__epsilon': np.logspace(-2, 2, 5),
        }
    },
    'RANDOM_FOREST': {
        'model': RandomForestRegressor(
            criterion='squared_error',
            n_jobs=configs.N_JOBS,
            random_state=configs.SEED,
            verbose=False,
            ),
        'params': {
            'model__regressor__n_estimators': [100, 250, 500],
            'model__regressor__max_depth': [3, 5, 10, None],
            'model__regressor__max_features': [0.3, 0.5, 0.7, None],
            'model__regressor__min_samples_leaf': [5, 10, 25, 50],
            }
    },
    'GBM': {
        'model': GradientBoostingRegressor(
            loss='squared_error',
            random_state=configs.SEED,
            verbose=False,
            ),
        'params': {
            'model__regressor__n_estimators': [50, 100, ],
            'model__regressor__learning_rate': np.logspace(-3, 3, num=5),
            'model__regressor__max_depth': [3, 5, 10],
            'model__regressor__max_features': [0.3, 0.7, None],
            'model__regressor__min_samples_leaf': [1, 10],
            'model__regressor__min_samples_split': [2],
            }
    },
    'MLP': {
        'model': MLPRegressor(
            solver='adam',
            n_iter_no_change=10,
            random_state=configs.SEED,
            verbose=False,
            ),
        'params': {
            'model__regressor__activation': ['relu', 'logistic', 'tanh'],
            'model__regressor__max_iter': [250, 500, 1000, 2000],
            'model__regressor__hidden_layer_sizes': [(5,), (10,), (20,), (3,3), (5,5), (5,2), (3,2), (4,2),],
            'model__regressor__learning_rate_init': np.logspace(-3, 3, 7),
            }
    },
}