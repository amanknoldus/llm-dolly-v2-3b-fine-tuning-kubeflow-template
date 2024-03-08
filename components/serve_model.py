from kfp.v2.components.component_decorator import component
from kfp.v2.dsl import Artifact, Output, Model
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform'
    )
)
def serve_model_component(
        project_id: str,
        location: str,
        staging_bucket: str,
        serving_image_uri: str,
        model_display_name: str,
        component_execution: bool,
        service_account: str,
        save_model_details_bucket: str,
        model_details_file_name: str,
        vertex_endpoint: Output[Artifact],
        vertex_model: Output[Model],
        machine_type: str = 'e2-highmem-8',
):
    """
    Function to upload model to model registry,
    serve model to an endpoint.
    @project_id: Project Unique ID.
    @location: specific region project located at.
    @staging_bucket: GCS Bucket used for model staging.
    @serving_image_uri: Serving Docker Image Path over GCP Artifact Registry.
    @model_display_name: Model display name
    @vertex_endpoint: created endpoint address
    @vertex_model: Model located at model registry
    @model_details:
    """
    from google.cloud import aiplatform
    from google.cloud import storage
    import logging
    import json
    import os

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        if not component_execution:
            logging.info("Component execution: serve model execution is bypassed")
        else:
            logging.info(f"Task: Initiating aiplatform for project: {project_id} & {location}")
            aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

            logging.info("Task: Uploading model to model registry")
            model = aiplatform.Model.upload(display_name=model_display_name,
                                            location=location,
                                            serving_container_image_uri=serving_image_uri,
                                            serving_container_ports=[8080]
                                            )

            logging.info("Task: Uploaded Model to Model Registry Successfully")

            logging.info("Task: Deploying Model to an endpoint")
            endpoint = model.deploy(machine_type=machine_type,
                                    min_replica_count=1,
                                    max_replica_count=2,
                                    accelerator_type=None,
                                    accelerator_count=None,
                                    service_account=service_account)
            logging.info(endpoint)

            vertex_endpoint.uri = endpoint.resource_name
            vertex_model.uri = model.resource_name

            logging.info("Task: Uploaded Model to an Endpoint Successfully")

            logging.info("Task: Extracting model id and endpoint id")
            deployed_display_name = f"{model_display_name}_endpoint"
            deployed_model_id = model.resource_name.split("/")[-1]
            endpoint_id = endpoint.resource_name.split("/")[-1]

            logging.info("Task: Appending ID's to the dictionary")
            model_details = {
                "deployed_display_name": deployed_display_name,
                "endpoint_id": endpoint_id,
                "deployed_model_id": deployed_model_id
            }

            logging.info(f"Task: Dumping model details to a json file as: {model_details_file_name}")
            with open(model_details_file_name, "w") as file:
                json.dump(model_details, file)

            logging.info("Task: Making client connection to save model details to bucket")
            client = storage.Client()
            bucket = client.get_bucket(save_model_details_bucket)

            blob = bucket.blob(model_details_file_name)

            logging.info(f"Task: Uploading model details to GCS Bucket: {save_model_details_bucket}")
            blob.upload_from_filename(model_details_file_name)

            logging.info("Task: Removing model details files from local environment")
            os.remove(model_details_file_name)

    except Exception as e:
        logging.error("Failed to Deployed Model To an Endpoint! Task: (serve_model_component)")
        raise e
