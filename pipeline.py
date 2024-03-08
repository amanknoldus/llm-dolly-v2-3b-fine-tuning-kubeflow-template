import logging
from kfp.v2 import compiler
import kfp

from components.process_data import process_data
from components.serve_model import serve_model_component
from components.train_model import fine_tune_model
from components.upload_model import upload_container
from constants import (PIPELINE_DESCRIPTION, PIPELINE_NAME, PIPELINE_ROOT_GCS, ORIGINAL_MODEL_NAME, \
                       SAVE_MODEL_BUCKET_NAME, REGION, DATASET_BUCKET, MODEL_DISPLAY_NAME, SERVING_IMAGE, \
                       STAGING_BUCKET, COMPONENT_EXECUTION, DATASET_NAME, SERVING_IMAGE_TRIGGER, SERVICE_ACCOUNT_ML,
                       DEPLOYED_MODEL_DETAILS_FILE,
                       PIPELINE_JSON_FILE)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name=PIPELINE_NAME,
                  description=PIPELINE_DESCRIPTION,
                  pipeline_root=PIPELINE_ROOT_GCS)
def pipeline(
        project_id: str,
        job_id: str
):
    """Dataset Processing"""
    process_data_task = process_data(DATASET_BUCKET, DATASET_NAME).set_display_name("Data_Processing")

    """Fine Tune Model Pipeline"""
    train_model_task = fine_tune_model(process_data_task.outputs["dataset"],
                                       ORIGINAL_MODEL_NAME,
                                       SAVE_MODEL_BUCKET_NAME,
                                       COMPONENT_EXECUTION) \
        .after(process_data_task) \
        .set_display_name("Dolly Fine Tuning") \
        .set_cpu_request("8") \
        .set_memory_limit("32G")

    """Upload model package"""
    upload_model_task = upload_container(project_id=project_id,
                                         trigger_id=SERVING_IMAGE_TRIGGER,
                                         component_execution=COMPONENT_EXECUTION) \
        .after(train_model_task) \
        .set_display_name("Model_Upload")

    """Serve Model To Endpoint"""
    serve_model_component(project_id,
                          REGION,
                          STAGING_BUCKET,
                          SERVING_IMAGE,
                          MODEL_DISPLAY_NAME,
                          COMPONENT_EXECUTION,
                          SERVICE_ACCOUNT_ML,
                          save_model_details_bucket=DATASET_BUCKET,
                          model_details_file_name=DEPLOYED_MODEL_DETAILS_FILE) \
        .after(upload_model_task) \
        .set_display_name("Serve_Model") \
        .set_cpu_request("8") \
        .set_memory_limit("32G")


def compile_pipeline(pipeline_template_name=f'{PIPELINE_JSON_FILE}'):
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_template_name
    )
    return None


if __name__ == "__main__":
    compile_pipeline()
