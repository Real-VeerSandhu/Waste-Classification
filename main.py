import base64
import io
import numpy as np
import tensorflow as tf 

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from PIL import Image

from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app=app)

def get_model():
    global model 
    model = tf.keras.models.load_model('models/resnet50_gar0.h5')
    global labels
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    print(' * Model loaded!')

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((256,256))
    array1 = np.array(image.getdata())
    img_np_array = np.reshape(array1, (256,256,3))
    final_image = np.expand_dims(img_np_array, axis=0)
    return final_image

print(' * Loading model...')
get_model()

@app.route('/predict', methods=['POST'])
def predict():

    message = request.get_json(force=True)
    encoded = message['image'] # string
    decoded = base64.b64decode(encoded) # bytes
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image) 

    prediction = model.predict(processed_image).tolist()
    output = labels[np.argmax(prediction[0])]

    print('->>> ', output)
    response = {
        'prediction': output
    }

    return jsonify(response)

app.run(host='0.0.0.0', port=5000)