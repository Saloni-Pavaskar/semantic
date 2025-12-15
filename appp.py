from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__, template_folder='Templates')  # Ensure this points to 'Templates'

# Initialize the pre-trained RoBERTa model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        review = request.form["review"]
        encoded_text = tokenizer(review, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        sentiment = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    return render_template("indexx.html", sentiment=sentiment)  # Use 'indexx.html' instead

if __name__ == "__main__":
    app.run(debug=True)


