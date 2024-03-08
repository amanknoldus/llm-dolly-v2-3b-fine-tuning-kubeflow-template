from google.cloud import aiplatform
from datetime import datetime
import os
from constants import REGION, SERVICE_ACCOUNT_ML, PIPELINE_NAME, PROJECT_ID, PIPELINE_JSON_BUCKET, PIPELINE_JSON_FILE


def run_pipeline_job(
        sync: bool = False, *,
        pipeline_template_name: str = f'gs://{PIPELINE_JSON_BUCKET}/{PIPELINE_JSON_FILE}',
        cleanup_compiled_pipeline: bool = False,
        enable_caching: bool = False,
) -> aiplatform.PipelineJob:
    job_id = f'{PIPELINE_NAME}-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    experiment_name = f'{PIPELINE_NAME}-kubeflow'

    params = dict(
        project_id=PROJECT_ID,
        job_id=job_id,
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)

    pipeline_job = aiplatform.PipelineJob(
        project=PROJECT_ID,
        location=REGION,
        display_name=PIPELINE_NAME,
        template_path=pipeline_template_name,
        parameter_values=params,
        job_id=job_id,
        enable_caching=enable_caching,
    )

    try:
        pipeline_job.submit(service_account=SERVICE_ACCOUNT_ML, experiment=experiment_name)
        if sync:
            pipeline_job.wait()
    finally:
        if cleanup_compiled_pipeline:
            os.remove(pipeline_template_name)

    return pipeline_job


if __name__ == "__main__":
    run_pipeline_job()
