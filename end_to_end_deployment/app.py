from flask import Flask, render_template, request
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("models/churn_prediction_model.pkl", "rb"))
with open("models/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/predict-form")
def predict_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract numeric inputs
        tenure = float(request.form["tenure"])
        citytier = float(request.form["citytier"])
        warehousetohome = float(request.form["warehousetohome"])
        hourspendonapp = float(request.form["hourspendonapp"])
        numberofdeviceregistered = float(request.form["numberofdeviceregistered"])
        satisfactionscore = float(request.form["satisfactionscore"])
        numberofaddress = float(request.form["numberofaddress"])
        complain = float(request.form["complain"])
        orderamounthikefromlastyear = float(request.form["orderamounthikefromlastyear"])
        couponused = float(request.form["couponused"])
        ordercount = float(request.form["ordercount"])
        daysincelastorder = float(request.form["daysincelastorder"])
        cashbackamount = float(request.form["cashbackamount"])

        # Categorical inputs
        gender = request.form["gender"]
        maritalstatus = request.form["maritalstatus"]

        # Prepare feature array
        x = np.zeros(len(data_columns))
        features = [
            tenure, citytier, warehousetohome, hourspendonapp,
            numberofdeviceregistered, satisfactionscore, numberofaddress,
            complain, orderamounthikefromlastyear, couponused, ordercount,
            daysincelastorder, cashbackamount
        ]
        x[:13] = features

        gender_col = f"gender_{gender.lower()}"
        marital_col = f"maritalstatus_{maritalstatus.lower()}"

        if gender_col in data_columns:
            x[data_columns.index(gender_col)] = 1
        if marital_col in data_columns:
            x[data_columns.index(marital_col)] = 1

        # Prediction and probability
        prediction = model.predict([x])[0]
        probabilities = model.predict_proba([x])[0]
        churn_prob = probabilities[1]  # 1 usually represents churn class

        # Determine labels correctly
        if prediction == 1:
            risk = "High Risk"
            risk_status = "Churn"
        else:
            risk = "Low Risk"
            risk_status = "Retain"

        return render_template("result.html", data={
            "risk": risk,
            "risk_status": risk_status,
            "predict_probability": round(churn_prob * 100, 2)
        })

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    print("🚀 ChurnAI running on http://localhost:5000")
    app.run(debug=True)
