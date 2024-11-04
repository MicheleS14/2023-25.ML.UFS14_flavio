import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler    

#Import the data into a dataframe
def model_run(
    path_to_dataset: str,
    model_dir: str,
):
    try:
        df = pd.read_csv(path_to_dataset)

        le = preprocessing.LabelEncoder()
        le.fit(df['diagnosis'])
        df['diagnosis'] = le.transform(df['diagnosis'])

        X = df[df.columns[2:-1]]
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(15, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')

        model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_test, y_test))

        losses = pd.DataFrame(model.history.history)
        # losses.plot()
        
        model.save_weights(rf'{model_dir}/my_model_weights.weights.h5')

        return 'Ha funzionato'
    
    except Exception as e:
        return str(e)

model_dir = os.environ['SM_MODEL_DIR']

print('#####################################')

input_dir = os.environ['SM_INPUT_DIR']
output = model_run(
    rf"{input_dir}/data/training/cancer_data.csv",
    model_dir
)
    
with open(model_dir + '/output_model.txt', 'w') as f:
    f.write(output)
    
output_dir = os.environ['SM_OUTPUT_DIR']
with open(output_dir + '/output.txt', 'w') as f:
    f.write('Ciao sono i log del training')