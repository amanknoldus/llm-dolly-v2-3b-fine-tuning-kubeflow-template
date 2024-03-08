from pathlib import Path

path = Path(__file__).resolve().parent

"""Below defined variable with there values, please change it accordingly"""

# Project id
PROJECT_ID = "your-project_id"

# Project Region
REGION = "your_project_region"

# Service account having all necessary rights.
SERVICE_ACCOUNT_ML = "your_service_account"

# Deploy model display name
MODEL_DISPLAY_NAME = "your_model_name"

# Pipeline display name
PIPELINE_NAME = "your-pipeline-name"

# Pipeline description
PIPELINE_DESCRIPTION = "your_pipeline_description"

# Pipeline root gcs bucket where the files will be saved during pipeline execution
PIPELINE_ROOT_GCS = f"gs://{PROJECT_ID}-kubeflow-pipeline"

# Base image name
BASE_IMAGE_QUALIFIER = "your_base_image_name"
# Base image tag
BASE_IMAGE_TAG = "your_base_image_tag as: 0.0.1"
# Base docker image complete path and name
ARTIFACT_REGISTRY = "your_artifact_registry"
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY}/{BASE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"

# Serve image name
SERVE_IMAGE_QUALIFIER = "your_serve_image_name"
# Serve image tag
SERVING_IMAGE_TAG = "your_serve_image_tag as: 0.0.1"
# Serve model docker image complete path and name
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY}/{SERVE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"

# Bucket where the workflow details will be stored while the pipeline execution
STAGING_BUCKET = "gs://{you_staging_bucket_name}/"

PIPELINE_JSON_BUCKET = "your_pipeline_file_bucket_name"
PIPELINE_JSON_FILE = "your_pipeline_file_name.json"

SAVE_MODEL_BUCKET_NAME = "your_bucket_name_to_save_model"

DATASET_BUCKET = "your_dataset_bucket"
DATASET_NAME = "your_dataset_name"

SERVING_IMAGE_TRIGGER = "your_serving_image_trigger_id"

COMPONENT_EXECUTION = True

ORIGINAL_MODEL_NAME = "databricks/dolly-v2-3b"

DEPLOYED_MODEL_DETAILS_FILE = "model_details.json"
