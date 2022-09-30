from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import sklearn

app = Flask(__name__, template_folder='web')

@app.route('/')
def company():
    return render_template("home.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)        
    loaded_model = joblib.load("joblib_lr_Model.pkl")
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = round(float(ValuePredictor(to_predict_list)), 2)
        return render_template("home.html", result=result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

