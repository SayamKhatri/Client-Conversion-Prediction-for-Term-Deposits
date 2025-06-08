from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn==1.3.2', 'joblib', 'google-cloud-aiplatform', 'numpy==1.24.3'],
    base_image= "python:3.10"
    # output_component_file='components/evaluate_rf/Eval_rf_model.yml'
           
)
def evaluate_rf(
    test_rf : Input[Dataset],
    model_rf : Input[Model],
    metrics_rf : Output[Artifact],
    project_id: str,
    region: str,
    experiment_name: str,
    run_name : str
                
)->None:

  import pandas as pd 
  from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
  import joblib
  import os 
  import json
  from google.cloud import aiplatform
  import pickle

  test_df = pd.read_csv(test_rf.path + '.csv')
  X_test = test_df.drop('y', axis = 1)
  y_test = test_df['y']

  with open(os.path.join(model_rf.path, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)


#   model = joblib.load(os.path.join(model_rf.path, 'model.joblib'))
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]


  metrics = {
      "accuracy": accuracy_score(y_test, y_pred),
      "precision": precision_score(y_test, y_pred),
      "recall": recall_score(y_test, y_pred),
      "auc": roc_auc_score(y_test, y_proba)
  }

  os.makedirs(metrics_rf.path, exist_ok=True)
  with open(os.path.join(metrics_rf.path, 'metrics_rf.json'), 'w') as f:
      json.dump(metrics, f, indent=4)

  print("Random Forest model evaluation complete.")
  print("Metrics:", metrics)


  aiplatform.init(project=project_id, location=region, experiment=experiment_name)

  with aiplatform.start_run(run=run_name) as run:

      run.log_params({
          'Model_Type': 'RandomForest',
          'Source': 'Kubeflow'
      })

      run.log_metrics(metrics)

