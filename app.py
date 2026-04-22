from flask import Flask, request, render_template
import pickle


app = Flask(__name__)


model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    data = vectorizer.transform([news])
    prediction = model.predict(data)[0]


    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"
    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)