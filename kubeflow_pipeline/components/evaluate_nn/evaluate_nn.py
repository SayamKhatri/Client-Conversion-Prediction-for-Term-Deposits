from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn', 'tensorflow', 'google-cloud-aiplatform'],
    base_image= "python:3.10"
    # output_component_file='components/evaluate_nn/eval_nn_model.yml'
           
)
def evaluate_nn(
    test_nn : Input[Dataset],
    model_nn : Input[Model],
    metrics_nn : Output[Artifact],
    project_id: str,
    region: str,
    experiment_name: str,
    run_name: str 
                
)->None:

  import pandas as pd 
  import tensorflow as tf 
  from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
  import os 
  import json
  from google.cloud import aiplatform
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import TFSMLayer

  test_df = pd.read_csv(test_nn.path + '.csv')
  X_test = test_df.drop('y', axis=1)
  y_test = test_df['y']




  X_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)

  model = TFSMLayer(model_nn.path, call_endpoint="serving_default")

 
  raw_output = model(X_tensor)
  if isinstance(raw_output, dict):
      y_proba = list(raw_output.values())[0].numpy().flatten()
  else:
      y_proba = raw_output.numpy().flatten()


  y_pred = (y_proba > 0.5).astype(int)
  print(" ✅ Precition Done")

  metrics = {
      "accuracy": accuracy_score(y_test, y_pred),
      "precision": precision_score(y_test, y_pred),
      "recall": recall_score(y_test, y_pred),
      "auc": roc_auc_score(y_test, y_proba)
  }


  os.makedirs(metrics_nn.path, exist_ok=True)
  with open(os.path.join(metrics_nn.path, 'metrics_nn.json'), 'w') as f:
      json.dump(metrics, f, indent=4)


  aiplatform.init(project=project_id, location=region, experiment=experiment_name)

  with aiplatform.start_run(run=run_name) as run:

      run.log_params({
          'Model_Type': 'NeuralNetwork',
          'Source': 'Kubeflow'
      })

      run.log_metrics(metrics)



  print("Neural Network model evaluation complete.")
  print("Metrics:", metrics)

