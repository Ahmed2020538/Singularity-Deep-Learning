from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/CNN_Model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route("/" , methods=["GET"])
def home():
    return render_template('index.html')

classes = {1:'buildings',
           2:'forest',
           3:'glacier',
           4: 'mountain',
           5: 'sea',
           6: 'street'}



@app.route("/Predict" , methods =["POST"])
def prediction():
    global COUNT

    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))
    img_arr = Image.open('static/{}.jpg'.format(COUNT))

    img = np.resize(img_arr, (100, 100))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    normalized = img
    reshaped = np.reshape(normalized, (1, 100, 100, 1))
    mg_arr = np.array(reshaped)
    result = model.predict_classes(mg_arr)[0]

    Class = classes[result +1]
    COUNT += 1

    return render_template('prediction.html', Class = Class)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

