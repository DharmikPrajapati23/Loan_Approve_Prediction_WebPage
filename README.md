# Loan Approval Prediction Model

This repository contains a machine learning-based web application for predicting **loan approval probabilities**. The project leverages **catboost**, a highly efficient gradient boosting algorithm, to predict whether a loan will be approved or not based on several input features. The application is hosted on [Render](https://loan-approve-prediction-webpage.onrender.com) for public accessibility.

---

## Features

The model uses the following features to predict the probability of loan approval:

1. **person_age**: Age of the applicant.
2. **person_income**: Annual income of the applicant.
3. **person_home_ownership**: Type of home ownership (e.g., Rent, Own, Mortgage).
4. **person_emp_length**: Length of employment in years.
5. **loan_intent**: Purpose of the loan (e.g., Education, Medical, Business).
6. **loan_grade**: Credit grade assigned to the applicant.
7. **loan_amnt**: Loan amount requested.
8. **loan_int_rate**: Interest rate of the loan.
9. **cb_person_default_on_file**: History of default on file (Yes/No).
10. **cb_person_cred_hist_length**: Length of credit history in years.

---

### Key Functionalities

- **Prediction**: The model predicts the **probability** of loan approval based on input data.
- **User-Friendly Interface**: A web interface for users to input their details and receive loan approval probabilities.
- **Deployment**: The web application is deployed on Render for seamless accessibility.

---

## Technology Stack

- **Machine Learning Framework**: catboost
- **Web Framework**: Flask
- **Deployment Platform**: Render
- **Programming Language**: Python
- **Frontend**: HTML, CSS

---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/<your-username>/Loan_Approve_Prediction_WebPage/tree/master
cd Loan_Approve_Prediction_WebPage
```
### Install Dependencies
Use pip to install the required dependencies:
```bash
pip install -r requirements.txt
```

### Run the Application Locally
Start the Flask server:
```bash
python app.py
```

---

Visit http://localhost:5000 in your browser to access the web application.

---

## Usage Instructions
1. Open the hosted application link or run it locally.
2. Input your details into the provided fields:
3. Click the "Predict" button.
4. The app will display the probability of loan approval.

---

### Licence
This project is licensed under the MIT License. See the LICENSE file for more details.

---

### Acknowledgments
- catboost Documentation
- Flask Framework
- Render Deployment

---
