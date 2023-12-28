from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open('gnbmodel.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Load the scaler from the pickle file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'CreditScore': float(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'Tenure': float(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': float(request.form['NumOfProducts']),
            'HasCrCard': int(request.form['HasCrCard']),
            'IsActiveMember': int(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }

        # Preprocess the user input
        user_df = pd.DataFrame([user_input])
        user_df['Geography_France'] = 1 if user_input['Geography'] == 'France' else 0
        user_df['Geography_Germany'] = 1 if user_input['Geography'] == 'Germany' else 0
        user_df['Geography_Spain'] = 1 if user_input['Geography'] == 'Spain' else 0
        user_df['Gender_Female'] = 1 if user_input['Gender'] == 'Female' else 0
        user_df['Gender_Male'] = 1 if user_input['Gender'] == 'Male' else 0

        # Drop the original Geography and Gender columns
        user_df = user_df.drop(['Geography', 'Gender'], axis=1)

        # Scale numerical features
        user_df_scaled = scaler.transform(user_df)

        # Use the trained model to predict the outcome
        user_prediction = rf_model.predict(user_df_scaled)

        return render_template('result.html', prediction=user_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
