from google.cloud import aiplatform
from google.cloud import storage
from constants import PROJECT_ID, REGION, DATASET_BUCKET, DEPLOYED_MODEL_DETAILS_FILE
import json
import os

from utils.helper_functions import get_log

logging = get_log()


def undeploy_model_from_endpoint(names):
    endpoints = aiplatform.Endpoint.list()
    for i in endpoints:
        if str(i.display_name) == names:
            i.undeploy_all()


def delete_endpoint_sample(
        project: str,
        endpoint_id: str,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
        timeout: int = 300,
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
    name = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.delete_endpoint(name=name)
    print("Long running operation:", response.operation.name)
    delete_endpoint_response = response.result(timeout=timeout)
    print("delete_endpoint_response:", delete_endpoint_response)


def delete_model_sample(model_id: str, project: str, location: str):
    aiplatform.init(project=project, location=location)
    model = aiplatform.Model(model_name=model_id)
    model.delete()


def delete_model_from_deployment(
        project: str,
        region: str,
        model_details_bucket: str,
        model_details_file_name: str,
):
    client = storage.Client()

    bucket = client.get_bucket(model_details_bucket)
    blob = bucket.blob(model_details_file_name)
    blob.download_to_filename(model_details_file_name)

    with open(model_details_file_name, "r") as json_file:
        data = json.load(json_file)

    deployed_display_name = data["deployed_display_name"]
    endpoint_id = data["endpoint_id"]
    deployed_model_id = data["deployed_model_id"]

    print(f"Deployed Model Details: deployed_display_name: {deployed_display_name}, endpoint_id: {endpoint_id}, "
          f"deployed_model_id: {deployed_model_id}")
    undeploy_model_from_endpoint(deployed_display_name)
    delete_endpoint_sample(project, endpoint_id)
    delete_model_sample(deployed_model_id, project, region)

    logging.info("Task: Removing model details files from local environment")
    os.remove(model_details_file_name)

    return "Model Undeployed Successfully"


delete_model_from_deployment(PROJECT_ID,
                             REGION,
                             DATASET_BUCKET,
                             DEPLOYED_MODEL_DETAILS_FILE)
