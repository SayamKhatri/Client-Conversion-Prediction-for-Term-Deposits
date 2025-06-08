from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs'],
    base_image= "python:3.10",
    output_component_file= 'getdata.yml'

)
def get_data(
    raw_gcs_url : str,
    raw_dataset : Output[Dataset]
) -> None:

  import pandas as pd 

  df = pd.read_csv(raw_gcs_url, sep=';')
  df['y'] = df['y'].map({'no': 0 , 'yes': 1})


  df.to_csv(raw_dataset.path + '.csv', index=False)
  print('Raw_Dataset_Saved at:',raw_dataset.path + '.csv')
