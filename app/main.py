#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64
import tensorflow as tf

# Initialize flask app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('app/mnist_classification.h5')

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('drawing.html')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to (28, 28)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    
    img=255-img
    np.set_printoptions(threshold=np.inf)

    # Expand to numpy array dimenstion to (1, 28, 28)
    img = np.expand_dims(gray_image, axis=0)
    img=cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img=np.array(img).reshape(-1,28,28,1)
    img=img.astype('float32')
    img=tf.keras.utils.normalize(img,axis=1)
    #print(img.shape)
    #print(img.dtype)

    try:
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template('drawing.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('drawing.html', response=str(e), canvasdata=canvasdata)


# In[ ]:




