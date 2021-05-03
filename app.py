from flask import Flask, request, jsonify
from fastai.vision.all import *
from fastai.vision.widgets import *
#from fastai.vision import open_image
from flask_cors import CORS,cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

path = Path()

#from fastai.basic_train import load_learner
learn_inf = load_learner(path/'export.pkl', cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


#classes = learn.data.classes


def predict_single(img_file):
    'function to take image and return prediction'
    lbl_pred.value = ''
    with out_pl: display(img_file.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img_file)
    #prediction = learn_inf.predict(open_image(img_file))
 #   probs_list = prediction[2].numpy()
    return prediction


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()
