
from analytics_translators import configs
from analytics_translators.model.features import create_preprocessor
from analytics_translators.model.params import model_objects
from analytics_translators.utils.export import save_pipeline

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, max_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import mlflow
import datetime
import shutil
import re
import numpy as np
import os
import warnings

from pathlib import Path

def main():
    
    if not Path(configs.DIR_MODEL_DEPLOY).exists():
        Path(configs.DIR_MODEL_DEPLOY).mkdir()

    df_all = pd.read_csv('output/df_all.csv')

    cols_target = [col for col in df_all.columns if col in configs.COLS_TARGET]
    cols_predictors = [col for col in df_all.columns if col not in configs.COLS_TARGET + configs.COLS_EXCLUDE]
    cols_predictors_bool = [col for col in cols_predictors if bool(re.match(configs.REGEX_BOOL_FEATURES, string=col))]
    cols_predictors_numeric = [col for col in cols_predictors if col not in cols_predictors_bool]

    if configs.BOOL_VERBOSE:
        print('cols_target:\n', cols_target)
        print('cols_predictors:\n', cols_predictors)

    df_all[cols_predictors_bool] = df_all[cols_predictors_bool].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        if configs.MLFLOW_BOOL:
            
            mlflow_path = Path(configs.MFLOW_DIR)
            
            # Delete previous experiment
            if mlflow_path.exists():
                shutil.rmtree(mlflow_path)
            os.mkdir(mlflow_path)

        for target in configs.COLS_TARGET:

            print(f'Modelling for {target}')

            # Use mlflow to track experiments
            if configs.MLFLOW_BOOL:
                # mlflow.warnings.filterwarnings(action='ignore')
                    
                ml_exp_name = configs.MLFLOW_EXPERIMENT_NAME + '_' + target
                ml_exp = mlflow.get_experiment_by_name(ml_exp_name)
                
                if ml_exp is not None:
                    # Delete previous experiment
                    mlflow.delete_experiment(ml_exp.experiment_id)

                experiment_id = mlflow.create_experiment(
                    ml_exp_name,
                    artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
                    tags={
                        "project": configs.PROJECT_NAME,
                        "target": target,
                        }
                )
                
                print(f'Created Experiment {ml_exp_name} | ID: {experiment_id}')

                mlflow.sklearn.autolog()
            
            # Split Data
            df = df_all.dropna(subset=target).copy()
            
            X = df[[col for col in df.columns if col in cols_predictors]]
            y = df[target]

            # Create training and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=configs.TEST_SIZE, 
                                                                random_state=configs.SEED)
            
            # Placeholder for best-performing model and metric
            best_model = None
            best_model_metric = np.inf
            
            # Train models
            for model_name in configs.MODELS:

                ts = datetime.datetime.now()
                
                if configs.MLFLOW_BOOL:
                    mlflow.start_run(
                        experiment_id=experiment_id, 
                        run_name=target+model_name+ts.strftime(r'_%Y%m%d_%H%m%s'), 
                        tags={
                            'model': model_name,
                            'target': target,
                            }
                        )
                    ml_run = mlflow.active_run()
                
                if configs.BOOL_VERBOSE:
                    print('_'*50)
                    print(model_name)
                    print(ts)

                # Pipeline for feature engineering and model
                pipeline = Pipeline(
                    steps=[
                        ('preprocessor', create_preprocessor(
                            col_bool=cols_predictors_bool,
                            col_numeric=cols_predictors_numeric
                            )),
                        ('pca', PCA(
                            n_components=np.max([
                                configs.PCA_MIN_COMPONENTS, 
                                int(len(cols_predictors) * configs.PCA_RATIO_TOTAL_COLUMNS)
                                ]),
                            random_state=configs.SEED
                            )),
                        ('model', TransformedTargetRegressor(
                            regressor=model_objects[model_name]['model'], 
                            func=np.log1p, 
                            inverse_func=np.expm1            
                            )),
                    ]
                )        
                
                # Hyperparameter tuning with k-fold cross-validation
                rs = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=model_objects[model_name]['params'],
                    n_iter=configs.SEARCH_ITERATIONS,
                    random_state=configs.SEED,
                    cv=configs.CV_FOLDS,
                    scoring={
                        'mse': make_scorer(score_func=mean_squared_error, squared=True, greater_is_better=False),
                        'rmse': make_scorer(score_func=mean_squared_error, squared=False, greater_is_better=False),
                        'mae': make_scorer(score_func=mean_absolute_error, greater_is_better=False),
                        'max_error': make_scorer(score_func=max_error, greater_is_better=False),
                        'r2': make_scorer(score_func=r2_score, greater_is_better=True),
                    },
                    refit=configs.CV_METRIC,
                    return_train_score=False,
                    n_jobs=configs.N_JOBS,
                    )

                rs.fit(X_train, y_train)
                y_pred = rs.predict(X_test)

                rmse_test = mean_squared_error(y_test, y_pred, squared=False)
                mae_test = mean_absolute_error(y_test, y_pred)
                max_error_test = max_error(y_test, y_pred)
                r2_test = r2_score(y_test, y_pred)
                
                # Export model
                save_pipeline(
                    pipeline_to_persist=rs, 
                    file_path=Path(configs.DIR_MODEL_DEPLOY) / f'model_{target}_{model_name}.pkl'
                    )
                
                if rmse_test < best_model_metric:
                    
                    best_model = model_name
                    best_model_metric = rmse_test
                    
                    # Export model
                    save_pipeline(
                        pipeline_to_persist=rs, 
                        file_path=Path(configs.DIR_MODEL_DEPLOY) / f'model_{target}_best.pkl'
                        )
                
                if configs.BOOL_VERBOSE:
                    print(f'RESULTS FOR {target} WITH {model_name}')
                    print("RMSE\n\tTrain: {:.5f}\n\tTest:  {:.5f}".format(-rs.cv_results_['mean_test_rmse'][rs.best_index_], rmse_test))
                    print("MAE\n\tTrain: {:.5f}\n\tTest:  {:.5f}".format(-rs.cv_results_['mean_test_mae'][rs.best_index_], mae_test))
                    print("MAX ERROR\n\tTrain: {:.5f}\n\tTest:  {:.5f}".format(-rs.cv_results_['mean_test_max_error'][rs.best_index_], max_error_test))
                    print("R-SQUARED\n\tTrain: {:.5f}\n\tTest:  {:.5f}".format(rs.cv_results_['mean_test_r2'][rs.best_index_], r2_test))

                    print("Best Parameters for {} model:\n {}".format(model_name, rs.best_params_))

                    tts = ts.now() - ts
                    print(f'Best model: {best_model}')
                    print(f'Training time {round(tts.total_seconds())}s')
                    
                
                if configs.MLFLOW_BOOL:
                    
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("cv_folds", configs.CV_FOLDS)
                    mlflow.log_param("PCA_MIN_COMPONENTS", configs.PCA_MIN_COMPONENTS)
                    mlflow.log_param("PCA_RATIO_TOTAL_COLUMNS", configs.PCA_RATIO_TOTAL_COLUMNS)
                    mlflow.log_param("CV_METRIC", configs.CV_METRIC)
                    mlflow.log_param("TEST_SIZE", configs.TEST_SIZE)
                    mlflow.log_param("SEARCH_ITERATIONS", configs.SEARCH_ITERATIONS)
                    mlflow.log_param("seed", configs.SEED)
                    mlflow.log_param("best_params", rs.best_params_)
                    
                    mlflow.log_metric('rmse_test', rmse_test)
                    mlflow.log_metric('mae_test', mae_test)
                    mlflow.log_metric('max_error_test', max_error_test)
                    mlflow.log_metric('r2_test', r2_test)
                    mlflow.sklearn.log_model(rs, 'model')
                    mlflow.end_run()


if __name__ == '__main__':
    main()