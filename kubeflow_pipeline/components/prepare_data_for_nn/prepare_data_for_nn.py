from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn'],
    base_image= "python:3.10",
    output_component_file= 'prepare_data_for_nn.yml'
)

def prepare_data_for_nf(
    raw_dataset : Input[Dataset],
    train_nn : Output[Dataset],
    test_nn : Output[Dataset],
    scaler_obj : Output[Artifact]
                        
)-> None:

  import pandas as pd 
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  import pickle

  df = pd.read_csv(raw_dataset.path + '.csv')

  if "duration" in df.columns:
      df.drop(columns=["duration"], inplace=True)

  categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'poutcome']
  numerical_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

  df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
  df = df.astype('float32')
  df['y'] = df['y'].astype(int)
  

  scaler = StandardScaler()
  df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

  train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

  train_df.to_csv(train_nn.path + '.csv', index=False)
  test_df.to_csv(test_nn.path + '.csv', index=False)

  with open(scaler_obj.path + '.pkl', 'wb') as f:
    pickle.dump(scaler, f)

  print('NN data ready')
  print('Scaler obj saved at:', scaler_obj.path + '.pkl')


