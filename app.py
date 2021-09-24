import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
import tensorflow as tf
import re
import tensorflow_hub as hub

app = FastAPI()

@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="https://github.com/Nawod/NDS" />
        <input type="submit" />'''

def preProcess_data(text):
    new_text = re.sub('[^a-zA-z0-9\s]',' ',text)
    return new_text

def embedding(text):
    embedding = 'https://tfhub.dev/google/nnlm-en-dim128/2'
    url_hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable= True)
    vector = url_hub_layer([text])

    return vector

@app.post('/predict')

def predict(text:str = Form(...)):
    # clean_text = preProcess_data(text) #clean the text
    loaded_model = tf.keras.models.load_model('malicious_url_model.h5') #load the saved model
    embeded_text =  embedding(text)
    predictions = loaded_model.predict(embeded_text) #predict the text
    
    sentiment = (predictions > 0.5).astype(np.int) #calculate the index of max sentiment
    # sentiment = 0
   
    if sentiment==0:
         t_sentiment = 'bad' #set appropriate sentiment
    elif sentiment==1:
         t_sentiment = 'good'
    return { #return the dictionary for endpoint
         "INPUT": text,
         "PREDICTION": predictions,
         "OUTPUT" : t_sentiment
         
    }