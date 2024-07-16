import joblib
import re
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder as Inc
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


__version__="1.0.0"

BASE_DIR = 'E:/pj 1/app/model'

model = load_model(f'{BASE_DIR}/tf_m_1.0.0.h5', compile=False)
model.compile(loss='mean_squared_error', optimizer='adam')

features =['cut', 'color', 'clarity', 'carat_weight', 'cut_quality', 'fab','symmetry','polish','eye_clean','cutlet_size','cutlet_condition','depth_percent','table_percent','meas_length','meas_width','meas_depth','girdle_min','girdle_max','fluor_color','fluor_intensity','fancy_color_dominant_','fancy_color_secondary','fancy_color_overtone','fancy_color_intensity']

def predict_pipe(l):
    ex = l
    exdf = np.array(ex).reshape(1,-1)
    exdf = pd.DataFrame (exdf, columns=features)

    C = 0
    for i in exdf:
        if isinstance(ex[C], str):
            lnc = joblib.load(f'{BASE_DIR}/{i}.joblib')
            exdf[i]= lnc.transform(exdf[i])
        C+=1

    xtest = exdf [features]
    xtest = np.array(xtest.astype(np.float64))

    return model.predict(xtest) [0] [0]
