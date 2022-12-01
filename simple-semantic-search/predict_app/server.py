import flask
from flask import request
from sentence_transformers import SentenceTransformer
from predict import KNN_Predictor

app = flask.Flask(__name__)
app.config["DEBUG"] = True
predictor = None

@app.route('/', methods=['GET'])
def home():
    return 'You should make a POST request to /predict'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('text')
    print('Received text for prediction: ', data)
    predicted_label = do_prediction(data)
    print('Predicted label: ', predicted_label)
    ret_val = 'Predicted label: ' + str(predicted_label)
    return ret_val

def do_prediction(data):
    pred = predictor.predict(data)
    return pred

if __name__ == '__main__':
    model_path = 'distilbert-base-nli-mean-tokens'
    model = SentenceTransformer(model_path)
    predictor = KNN_Predictor(model)
    app.run(debug=True, host='0.0.0.0')
