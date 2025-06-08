from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['fsspec', 'gcsfs'],
    base_image='python:3.10'
    # output_component_file='components/select_best_model/select_best_model.yml'
)
def select_best_model(
    metrics_rf: Input[Artifact],
    metrics_nn: Input[Artifact],
    model_rf: Input[Model],
    model_nn: Input[Model],
    best_model: Output[Artifact],
    should_deploy: Output[Artifact],
    best_model_path: Output[Artifact],
    best_model_framework: Output[Artifact],
    deploy_thresholds: dict = {
        "auc": 0.70,
        "accuracy": 0.70
    },
    primary_metric: str = "auc"
) -> None:
    import os, json

    with open(os.path.join(metrics_rf.path, 'metrics_rf.json')) as f:
        rf_metrics = json.load(f)
    with open(os.path.join(metrics_nn.path, 'metrics_nn.json')) as f:
        nn_metrics = json.load(f)

    best = "rf" if rf_metrics[primary_metric] >= nn_metrics[primary_metric] else "nn"
    best_metrics = rf_metrics if best == "rf" else nn_metrics

    meets_thresholds = all(
        best_metrics.get(metric, 0) >= threshold
        for metric, threshold in deploy_thresholds.items()
    )

    os.makedirs(best_model.path, exist_ok=True)
    with open(os.path.join(best_model.path, 'best_model.txt'), 'w') as f:
        f.write(best)

    os.makedirs(should_deploy.path, exist_ok=True)
    with open(os.path.join(should_deploy.path, 'deploy.txt'), 'w') as f:
        f.write("True" if meets_thresholds else "False")

    os.makedirs(best_model_path.path, exist_ok=True)
    best_path = model_rf.path if best == "rf" else model_nn.path
    with open(os.path.join(best_model_path.path, 'path.txt'), 'w') as f:
        f.write(best_path)

    os.makedirs(best_model_framework.path, exist_ok=True)
    with open(os.path.join(best_model_framework.path, 'framework.txt'), 'w') as f:
        f.write("sklearn" if best == "rf" else "tensorflow")

    print("Best model:", best)
    print("Path:", best_path)
    print("Should deploy:", meets_thresholds)
