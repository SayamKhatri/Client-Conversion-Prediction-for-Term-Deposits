from kfp.v2 import dsl 
from kfp.v2.dsl import Artifact, Dataset, component, Output, pipeline, Input, Model

@component(
    packages_to_install=['pandas', 'fsspec', 'gcsfs', 'scikit-learn', 'tensorflow'],
    base_image= "python:3.10"
    # output_component_file='components/train_nn_model/train_nn_model.yml'
           
)
def train_nn_model(
    train_nn : Input[Dataset],
    model_nn : Output[Model]
                   
)->None:

  import pandas as pd 
  import tensorflow as tf 
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.callbacks import EarlyStopping 

  import os

  train_df = pd.read_csv(train_nn.path + '.csv')
  X_train = train_df.drop('y', axis=1)
  y_train = train_df['y']


  input_dim = X_train.shape[1] 

  model = Sequential([
      Dense(16, activation='relu', input_dim=input_dim),
      Dense(8, activation='relu'),
      Dense(1, activation='sigmoid')
                   
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
  

  callback = EarlyStopping(
      monitor = 'val_loss',
      patience = 2
  )

  history = model.fit(
      X_train, y_train,
      validation_split = 0.2,
      epochs=20,
      batch_size = 8,
      verbose=1,
      callbacks = [callback]
  )

  os.makedirs(model_nn.path, exist_ok=True)
  model.export(model_nn.path)

  # model.save(model_nn.path)



  print('NN saved succesfuly')