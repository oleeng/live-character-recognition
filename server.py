from flask import Flask, request
import json
import numpy as np
import tensorflow as tf

def getReadableLabel(label):
    if label<10:
        # number
        return chr(label+48)
    elif label<36:
        # uppercase
        return chr(label+55)
    else:
        # lowercase
        return chr(label+61)

model = tf.keras.models.load_model('saved_model/my_model')

app = Flask(__name__)

@app.route('/')
def index():
    return open("gui/main.html", "r").read()

@app.route('/recognize', methods=['POST'])
def recognize():
    data = np.array(json.loads(request.data)).reshape((28,28,1))
    # normalize
    data = data / 255
    # invert
    data = (data - 1) * (-1)
    # predict
    data = data.reshape((1, 28, 28, 1))
    pre = model.predict(data.T)[0]

    # extract result and build response
    best5 = np.flip(np.argsort(pre)[-5:])
    best5Percent = pre[best5]

    response = []

    for i in range(0, 5):
        response.append([getReadableLabel(best5[i]), best5Percent[i] * 100])

    return json.dumps(response)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
