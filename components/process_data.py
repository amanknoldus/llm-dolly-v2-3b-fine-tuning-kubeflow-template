from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'fsspec'
    )
)
def process_data(
        dataset_bucket: str,
        dataset_file_name: str,
        dataset: dsl.Output[dsl.Dataset],
        dataset_details: dsl.Output[dsl.Metrics]

):
    """
    Function to load dataset from gcs bucket and pass to next component as a parquet file path.
    @param dataset: parquet file dataset path
    @param dataset_bucket: bucket name where dataset is stored
    @param dataset_details: dataset dimension details
    @param dataset_file_name: name of the dataset
    """
    import logging
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.debug(f"Task: Getting dataset from gcs bucket: '{dataset_bucket}'")
        read_file = pd.read_csv(f"gs://{dataset_bucket}/{dataset_file_name}")
        train_df = pd.DataFrame(read_file)

        logging.debug("Task: Saving dataset to parquet file")
        train_df.to_parquet(dataset.path)

        logging.debug("Task: Getting  shape of dataset")
        shape_of_data = train_df.shape

        logging.debug("Task: Displaying dataset shape")
        dataset_details.log_metric("Dataset Dimensions:", shape_of_data)

    except Exception as e:
        logging.error("Failed to process data!")
        raise e
