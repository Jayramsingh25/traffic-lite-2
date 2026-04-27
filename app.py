from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    traffic = float(request.form["traffic"])
    speed = float(request.form["speed"])
    hour = float(request.form["hour"])

    features = np.array([[traffic, speed, hour]])
    prediction = model.predict(features)

    return render_template("index.html",
                           prediction_text=f"Green Signal Time: {prediction[0]:.2f} sec")

if __name__ == "__main__":
    app.run(debug=True)