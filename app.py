from flask import Flask, render_template, request, redirect, url_for
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('abstract.html')

@app.route('/abstract', methods=['GET', 'POST'])
def abstract():
    return render_template('abstract.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/dectreeprediction', methods=['GET', 'POST'])
def dectreeprediction():
    show = False;
    if request.method == 'POST':
            with open('dtree_pickle', 'rb') as f:
                dtree_pickle = pickle.load(f)

            snoring_rate = float(request.form['snoringrate-input'])
            respiration_rate = float(request.form['respirationrate-input'])
            bodytemp = float(request.form['bodytemp-input'])
            limb_movement = float(request.form['limbmovement-input'])
            blood_oxygen = float(request.form['bloodoxygen-input'])
            rem = float(request.form['rem-input'])
            sleeping_hours = float(request.form['sleepinghours-input'])
            heart_rate = float(request.form['heartrate-input'])

            data = np.array([[snoring_rate, respiration_rate, bodytemp, limb_movement, blood_oxygen, rem, sleeping_hours, heart_rate]])
            data = data.reshape(1, -1)
            dtree_prediction = dtree_pickle.predict(data)
            show = True
            final_dtree_prediction = [show, dtree_prediction]
            return render_template('prediction.html', final_dtree_prediction = final_dtree_prediction)
    else:
        return render_template('prediction.html')


@app.route('/svmprediction', methods=['GET', 'POST'])
def svmprediction():
    show = False;
    if request.method == 'POST':
            with open('svm_pickle', 'rb') as f:
                svm_pickle = pickle.load(f)

            snoring_rate = float(request.form['snoringrate-input'])
            respiration_rate = float(request.form['respirationrate-input'])
            bodytemp = float(request.form['bodytemp-input'])
            limb_movement = float(request.form['limbmovement-input'])
            blood_oxygen = float(request.form['bloodoxygen-input'])
            rem = float(request.form['rem-input'])
            sleeping_hours = float(request.form['sleepinghours-input'])
            heart_rate = float(request.form['heartrate-input'])

            data = np.array([[snoring_rate, respiration_rate, bodytemp, limb_movement, blood_oxygen, rem, sleeping_hours, heart_rate]])
            data = data.reshape(1, -1)
            svm_prediction = svm_pickle.predict(data)
            show = True
            final_svm_prediction = [show, svm_prediction]
            return render_template('prediction.html', final_svm_prediction = final_svm_prediction)
    else:
        return render_template('prediction.html')


if __name__ == '__main__':
    app.run(use_reloader = True, debug = True)