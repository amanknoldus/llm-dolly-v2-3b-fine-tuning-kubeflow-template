import logging
from google.cloud.devtools import cloudbuild_v1

logging.basicConfig(level=logging.INFO)


def save_model(bucket_name: str, model_dir: str):
    """
    Function to save model files to the gcs_bucket.
    @param bucket_name: gcs_bucket name where model files has to be saved
    @param model_dir:local model files directory
    """
    import logging
    from google.cloud import storage
    import os

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.debug("Task: Initializing GCP Storage Client")
        client = storage.Client()

        logging.debug("Task: Getting Bucket")
        bucket = client.get_bucket(bucket_name)

        logging.debug(f"Task: Listing files present in the directory: {model_dir}")
        files = os.listdir(model_dir)

        logging.debug(f"Task: Iterating over each file present in the directory: {model_dir}")
        for file_name in files:
            file_path = os.path.join(model_dir, file_name)

            if os.path.isfile(file_path):
                logging.debug(f"Task: Creating blob object to upload the file to GCS Bucket: {bucket_name}")
                blob = bucket.blob(file_name)
                blob.upload_from_filename(file_path)

                logging.debug(f"Task: Uploaded file: {file_name} to {bucket_name} successfully")

    except Exception as e:
        logging.error(f"Some error occurred in uploading model files to the bucket nameL: {bucket_name}")
        raise e


def upload_model(get_project_id, get_trigger_id):
    logging.info("Task: Making Client Connection: ")
    cloud_build_client = cloudbuild_v1.CloudBuildClient()

    logging.info("Task: Triggering Cloud Build For Dolly Model Serving Container")
    response = cloud_build_client.run_build_trigger(project_id=get_project_id, trigger_id=get_trigger_id)

    logging.info(f"Cloud Build Trigger Execution Status: {response.running()}")
    trigger_details = response.metadata
    build_data = trigger_details.build
    log_path = build_data.log_url

    try:
        if response.result():
            logging.info("Cloud Build Successful")
            return True, log_path

    except Exception as error:
        logging.error("Some error occurred in cloud build execution")
        return error, log_path
