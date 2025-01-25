from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained pipeline model
model = pickle.load(open('pipeline.pkl', 'rb'))

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data from HTML page
    person_age = int(request.form['person_age'])
    person_income = int(request.form['person_income'])
    person_emp_length = float(request.form['person_emp_length'])
    loan_amnt = int(request.form['loan_amnt'])
    loan_int_rate = float(request.form['loan_int_rate'])
    cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
    person_home_ownership = request.form['person_home_ownership']
    loan_grade = request.form['loan_grade']
    cb_person_default_on_file = request.form['cb_person_default_on_file']
    loan_intent = request.form['loan_intent']
    
    # Create feature array in the order expected by the pipeline
    features = pd.DataFrame([[
        person_age, person_income, person_home_ownership, person_emp_length, 
        loan_intent, loan_grade, loan_amnt, loan_int_rate, 
        cb_person_default_on_file, cb_person_cred_hist_length 
    ]], columns=[
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
        'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ])
    # features = np.array([[ 
    #     person_age, person_income, person_home_ownership, person_emp_length, 
    #     loan_intent, loan_grade, loan_amnt, loan_int_rate, 
    #     cb_person_default_on_file, cb_person_cred_hist_length 
    # ]])

    # Predict using the pipeline model
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features)
    except Exception as e:
        return render_template('index.html', text=f"Error during prediction: {str(e)}")

    # # Predict using the pipeline model
    # prediction = model.predict(features)
    # probability = model.predict_proba(features)

    # Extract the probabilities for each class
    prob_approved = probability[0, 1] * 100  # Probability of loan approval
    prob_not_approved = probability[0, 0] * 100  # Probability of loan not being approved


    # Render the result in the HTML template with grammar corrected
    return render_template('index.html', 
                       text=f'<span class="probabilities">Your chance of loan approval is {prob_approved:.2f}%<br>'
                            f'And chance of not having the loan approved is {prob_not_approved:.2f}%</span>')


if __name__ == '__main__':
    app.run(debug=True)
