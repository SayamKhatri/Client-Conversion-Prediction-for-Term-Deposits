from kfp.v2.dsl import component, Input, Output, Artifact

@component(
    packages_to_install=['google-cloud-aiplatform'],
    base_image="python:3.10",
    output_component_file="components/deploy_model/deploy_model.yaml"
)
def deploy_model(
    registered_model: Input[Artifact],
    project_id: str,
    region: str,
    endpoint_uri: Output[Artifact],
    endpoint_display_name: str = "bank-model-endpoint",
    deployed_model_display_name: str = "bank-model-deployed",
    machine_type: str = "n1-standard-4"

):
    import os
    from google.cloud import aiplatform

    model_uri_file = os.path.join(registered_model.path, 'model_uri.txt')
    with open(model_uri_file, 'r') as f:
        model_resource_name = f.read().strip()


    aiplatform.init(project=project_id, location=region)


    model = aiplatform.Model(model_resource_name)


    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"', order_by='create_time desc')
    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        print(f"Created new endpoint: {endpoint.name}")


    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        traffic_split={"0": 100}
    )


    os.makedirs(endpoint_uri.path, exist_ok=True)
    endpoint_uri_file = os.path.join(endpoint_uri.path, 'endpoint_uri.txt')
    with open(endpoint_uri_file, 'w') as f:
        f.write(endpoint.resource_name)

    print(f"Model deployed to endpoint: {endpoint.resource_name}")
