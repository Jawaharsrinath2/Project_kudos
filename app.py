from flask import Flask, request, render_template_string
import pickle
import os
import csv
from model import TfidfNaiveBayes

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = None  # global lazy-loaded model

def get_model():
    global model

    if model is not None:
        return model

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            return model

    # Train model ONLY once, lazily
    model = TfidfNaiveBayes()

    with open("data.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model.train(row["review"], row["sentiment"])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


HTML = """
<h2>Sentiment Analysis (From Scratch)</h2>
<form method="post">
    <textarea name="text" rows="4" cols="50"></textarea><br><br>
    <button type="submit">Analyze</button>
</form>
{% if result %}
<h3>Sentiment: {{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        mdl = get_model()
        result = mdl.predict(text)
    return render_template_string(HTML, result=result)


