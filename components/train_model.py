from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'numpy',
        'fsspec',
        'pyarrow',
        'gcsfs',
        'google-cloud-storage',
        'datasets',
        'accelerate',
        'torch',
        'psutil',
        'transformers'
    )
)
def fine_tune_model(dataset_path: dsl.Input[dsl.Dataset],
                    model_name: str,
                    save_model_bucket_name: str,
                    component_execution: bool,
                    model_artifact_path: dsl.Output[dsl.Artifact]
                    ):
    """
    Function to perform model fine_tuning on custom dataset and save the trained model to gcs bucket
    @param dataset_path: parquet file dataset path
    @param model_name: original model name
    @param save_model_bucket_name: bucket name where model needs to be saved
    @param model_artifact_path: bucket path where model is saved and displayed as output in pipeline
    @param component_execution:
    """
    import logging
    from src import model

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        if not component_execution:
            logging.info("Component execution: model training is bypassed")
        else:
            logging.info("Started Model Training Task")
            model.fine_tune_model(dataset_path.path, model_name, save_model_bucket_name)

            logging.info("Task: Setting saved model directory bucket path")
            model_artifact_path.uri = f'https://storage.cloud.google.com/{save_model_bucket_name}'

    except Exception as e:
        logging.error("Failed to train model!")
        raise e
