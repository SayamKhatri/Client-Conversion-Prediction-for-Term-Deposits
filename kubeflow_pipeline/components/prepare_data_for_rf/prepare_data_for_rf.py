from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn'],
    base_image= "python:3.10"
    # output_component_file= 'components/prepare_data_for_rf/prepare_data_for_rf.yml'
)

def prepare_data_for_rf(
    raw_dataset : Input[Dataset],
    train_rf : Output[Dataset],
    test_rf : Output[Dataset],
    label_encoder_obj : Output[Artifact]
    
)-> None:

  import pandas as pd
  from sklearn.preprocessing import LabelEncoder 
  from sklearn.model_selection import train_test_split
  import pickle

  df = pd.read_csv(raw_dataset.path + '.csv')

  if 'duration' in df.columns:
    df.drop(columns=['duration'], inplace=True)

  
  categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

  label_encoders = {}
  for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 


  train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

  train_df.to_csv(train_rf.path + '.csv', index=False)
  test_df.to_csv(test_rf.path + '.csv', index=False)

  with open(label_encoder_obj.path + '.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


  print('Random Forest Data is Ready')
  print('Label Encoder artifact saved at:', label_encoder_obj.path + '.pkl')