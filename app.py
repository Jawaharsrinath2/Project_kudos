from flask import Flask, request, render_template_string
import pickle
import os
import csv
from model import TfidfNaiveBayes

app = Flask(__name__)

MODEL_PATH = "model.pkl"

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    model = TfidfNaiveBayes()

    with open("data.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model.train(row["review"], row["sentiment"])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

model = load_or_train_model()

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
        result = model.predict(text)
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
