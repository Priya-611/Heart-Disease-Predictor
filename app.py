from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app=Flask(__name__)
CORS(app)

with open('Heart Disease/heart_disease.pkl','rb') as f:
    model=pickle.load(f)

with open('Heart Disease/sex.pkl','rb') as f:
    sex_encoder=pickle.load(f)

with open('Heart Disease/chestPainType.pkl','rb') as f:
    chestPain_encoder=pickle.load(f)

with open('Heart Disease/restingECG.pkl','rb') as f:
    restingECG_encodeder=pickle.load(f)

with open('Heart Disease/exerciseAngina.pkl','rb') as f:
    ExerciseAgina_encoder=pickle.load(f)

with open('Heart Disease/stSlope.pkl','rb') as f:
    st_slopeencoder=pickle.load(f)






@app.route('/', methods=['GET'])
def index():
    return render_template('heart.html',prediction=None)




@app.route('/api/predict', methods=['POST'])
def api_predict():
    data=request.get_json()
    try:
        age=int(data['age'])
        cholestrol=int(data['cholestrol'])
        fastingBS=int(data['fastingBS'])
        oldpeak=float(data['oldpeak'])


        sex=sex_encoder.transform(data['sex'])[0]
        chestPain=chestPain_encoder.transform(data['chestPainType'])[0]
        restingECG=restingECG_encodeder.transform(data['restingECG'])[0]
        ExerciseAgina=ExerciseAgina_encoder.transform(data['exerciseAngina'])[0]
        st_slope=st_slopeencoder.transform(data['stSlope'])[0]

        features=np.array([age,sex,chestPain,cholestrol,fastingBS,restingECG,ExerciseAgina,oldpeak,st_slope])

        prediction=model.predict([features])[0]

        if prediction == 1:
            message = 'You have chances of getting a heart disease.'
        else:
            message = 'You have a healthy heart.'

        return jsonify({'prediction': int(prediction), 'message': message})

    except Exception as e:
        return jsonify({'error':str(e)}),400


if __name__=='__main__':
    app.run(debug=True)



