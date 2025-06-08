from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model
@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn', 'joblib'],
    base_image= "python:3.10"
    # output_component_file='train_rf_model.yml'
)
def train_rf_model(
    train_rf : Input[Dataset],
    model_rf : Output[Model]
                   
)-> None:
  import pandas as pd 
  from sklearn.ensemble import RandomForestClassifier
  import joblib
  import os
  import pickle

  train_df = pd.read_csv(train_rf.path + '.csv')
  X_train = train_df.drop('y', axis=1)
  y_train = train_df['y']


  model = RandomForestClassifier(n_estimators=100,random_state=42)
  model.fit(X_train, y_train)

  os.makedirs(model_rf.path, exist_ok=True)
  model_path = os.path.join(model_rf.path, 'model.joblib')
  if os.path.exists(model_path):
      os.remove(model_path)

  joblib.dump(model, model_path, protocol=4)


  joblib.dump(
    model, 
    os.path.join(model_rf.path, 'model.joblib'),
    protocol=4  
  )



  with open(os.path.join(model_rf.path, 'model.joblib'), 'rb') as f:
      unpickler = pickle._Unpickler(f)
      print("Pickle protocol used to save model:", unpickler.proto)


  print('RF model trained and saved at:', model_rf.path)