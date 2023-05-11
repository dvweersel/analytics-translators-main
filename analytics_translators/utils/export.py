import os
import joblib
from sklearn.pipeline import Pipeline
import typing as t


def save_pipeline(pipeline_to_persist: Pipeline, file_path: str) -> None:
    """Saves the pipeline to a pkl file

    Args:
        pipeline_to_persist (Pipeline): pipeline to save
        file_path (str): file path to save the file
    """
    
    joblib.dump(pipeline_to_persist, file_path)



def load_pipeline(file_path: str) -> Pipeline:
    """Load a pipeline saved as a pkl file

    Args:
        file_path (str): file path of file

    Returns:
        Pipeline: a scikit-learn Pipeline object
    """
    
    
    trained_model = joblib.load(filename=file_path)
    
    return trained_model


def remove_old_pipelines(files_to_keep: t.List[str], path_models: str) -> None:
    """
    Removes old model pipelines
    """
    do_not_delete = files_to_keep + ["__init__.py", 'training_data.csv']
    for model_file in os.listdir(path_models):
        
        path_file = os.path.join(path_models, model_file)
        
        if (model_file not in do_not_delete) and ('.png' not in path_file):
            os.remove(path_file) 