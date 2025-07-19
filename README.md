# Heart Disease Risk Prediction Web App

A machine learning-powered web application to predict the likelihood of heart disease based on user health indicators. This project combines a logistic regression model, a Flask backend, and an AI-generated frontend to deliver real-time predictions.

The main objective of this project is to demonstrate how machine learning models can be deployed in real-time applications and made accessible through simple web interfaces.

---


## Dataset

- **Source:** [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features Used:**
  - Age
  - Sex
  - Chest Pain Type
  - RestingBP
  - Cholesterol
  - Fasting Blood Sugar
  - Resting ECG
  - MaxHR
  - Exercise-Induced Angina
  - ST Depression (Oldpeak)
  - ST Slope
  - HeartDisease: output class [1: heart disease, 0: Normal]


---


##  Machine Learning Model

- **Model Used:** Logistic Regression
- **Preprocessing:** Label Encoding for categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`)
- **Performance:** ~84% accuracy on test data
- **Encoders and model** saved as `.pkl` files for use in the web application.


---


## Web Application

- **Backend:** Flask (Python)
- **Frontend:** AI-generated HTML (served using Jinja2 via Flask)
- **API Endpoint:** `POST /api/predict`  
  Accepts JSON input and returns prediction with a message.


---


## Project Structure

heart-disease-predictor/

├── Heart Disease/ `Saved model and encoders`

│ ├── heart.csv `Dataset`

│ ├── heart_disease.pkl `Trained logistic regression model`

│ ├── sex.pkl `Label encoder for 'Sex'`

│ ├── chestPainType.pkl `Label encoder for 'ChestPainType'`

│ ├── restingECG.pkl `Label encoder for 'RestingECG'`

│ ├── exerciseAngina.pkl `Label encoder for 'ExerciseAngina'`

│ └── stSlope.pkl `Label encoder for 'ST_Slope'`

│

├── templates/

│ └── heart.html `Frontend UI for prediction`

│

├── app.py `Flask backend code`

├── heart.py `Model training and preprocessing`

├── requirements.txt `Python dependencies`



---


## Project Workflow: How to Run Locally

1. **Clone the Repository**  
   Start by cloning the GitHub repository to your local machine. Open a terminal and run:


2. **Set Up a Virtual Environment (Optional but Recommended)**  
Create a virtual environment to manage dependencies:
- On Windows:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```


3. **Install Required Dependencies**  
Install all the necessary packages using pip:
pip install -r requirements.txt


4. **Train the Model (if not already trained)**  
If the model and encoder `.pkl` files do not exist, generate them by running:
`heart.py`

This will train the model using the dataset and save all required files into the `Heart Disease/` folder.


5. **Run the Flask Web App**  
Start the Flask development server by running:
`app.py`

After running this command, the app will be available at:
http://127.0.0.1:5000/


6. **Use the Web Interface**  
Open a browser and navigate to the above URL. You’ll see a form where you can input details and get a prediction.


7. **Use the API (Optional)**  
You can also make a POST request to `/api/predict` with JSON data using Postman or any HTTP client.


---


##  API Usage

**Endpoint:** `/api/predict`  
**Method:** POST  
**Content-Type:** `application/json`  


### Sample Request:
```json
{
"age": 54,
"sex": ["Female"],
"chestPainType": ["ATA"],
"cholestrol": 220,
"fastingBS": 1,
"restingECG": ["Normal"],
"exerciseAngina": ["No"],
"oldpeak": 1.8,
"stSlope": ["Flat"]
}

{
  "prediction": 1,
  "message": "You have chances of getting a heart disease."
}
```
