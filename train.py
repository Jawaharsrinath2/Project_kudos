import csv
import pickle
from model import TfidfNaiveBayes

model = TfidfNaiveBayes()

with open("data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        text = row["review"]
        label = row["sentiment"]
        model.train(text, label)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained from CSV and saved successfully")
