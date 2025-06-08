from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['google-cloud-aiplatform', 'fsspec', 'gcsfs'],
    base_image="python:3.10",
    output_component_file="register_model.yml"
)


def register_model(
    should_deploy: Input[Artifact],
    best_model_path: Input[Artifact],
    best_model_framework: Input[Artifact],
    project_id: str,
    region: str,
    registered_model: Output[Artifact],
    model_display_name: str = "bank-model"

                   
):
  import os 
  from google.cloud import aiplatform

  
  with open(os.path.join(should_deploy.path, 'deploy.txt'), 'r') as f:
    deploy_flag = f.read().strip()

  
  
  if deploy_flag.lower() != "true":
      print("Deployment skipped: Model did not meet thresholds.")
      return

  with open(os.path.join(best_model_path.path, "path.txt"), "r") as f:
      model_dir = f.read().strip()

  with open(os.path.join(best_model_framework.path, "framework.txt"), "r") as f:
      framework = f.read().strip()

  print(f"Registering model from path: {model_dir}")
  print(f"Framework: {framework}")



  aiplatform.init(project=project_id, location=region)

  model = aiplatform.Model.upload(
      display_name = model_display_name,
      artifact_uri = model_dir,

      serving_container_image_uri=(
          "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
          if framework == "sklearn" else
          "us-docker.pkg.dev/vertex-ai/prediction/tf-cpu.2-11:latest"
      )
    )                                
  
  
  model.wait()


  os.makedirs(registered_model.path, exist_ok=True)
  with open(os.path.join(registered_model.path, 'model_uri.txt'), 'w') as f:
      f.write(model.resource_name)

  print(f"Model registered: {model.resource_name}")