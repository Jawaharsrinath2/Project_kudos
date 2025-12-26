from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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
    app.run()
