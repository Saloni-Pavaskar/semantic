# Semantic Analysis Web App

A simple **Flask web application** for performing **sentiment analysis** on text using a pre-trained **RoBERTa model** (`cardiffnlp/twitter-roberta-base-sentiment`). Users can input text and get probability scores for **positive, neutral, and negative** sentiment.

## Features

- Input text and get sentiment scores.
- Uses Hugging Face's RoBERTa model for accurate sentiment prediction.
- Clean and responsive UI with custom styling (`style.css`) and template (`indexx.html`).

## Project Structure

semantic/
├── appp.py # Flask app
├── Templates/
│ └── indexx.html # HTML template
├── style.css # Custom CSS
└── README.md # Project documentation

Clone the repository:
```bash
git clone https://github.com/Saloni-Pavaskar/semantic.git
cd semantic
