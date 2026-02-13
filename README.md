🧠 Amazon Review Sentiment Analysis – NLP & Machine Learning

Analyzing customer sentiment from Amazon product reviews using Natural Language Processing and Machine Learning.

📌 Table of Contents

<a href="#overview">Overview</a>

<a href="#business-problem">Business Problem</a>

<a href="#dataset">Dataset</a>

<a href="#tools--technologies">Tools & Technologies</a>

<a href="#project-structure">Project Structure</a>

<a href="#data-cleaning--preparation">Data Cleaning & Preparation</a>

<a href="#exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>

<a href="#model-development">Model Development</a>

<a href="#model-evaluation">Model Evaluation</a>

<a href="#key-insights--findings">Key Insights & Findings</a>

<a href="#how-to-run-this-project">How to Run This Project</a>

<a href="#future-improvements">Future Improvements</a>

<a href="#author--contact">Author & Contact</a>

<h2><a class="anchor" id="overview"></a>Overview</h2>

This project performs sentiment classification on Amazon product reviews to categorize customer feedback into:

Positive

Neutral

Negative

The objective is to transform unstructured review text into meaningful business insights using NLP preprocessing techniques and supervised machine learning models.

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

E-commerce platforms rely heavily on customer feedback to:

Monitor product satisfaction

Identify recurring complaints

Improve product quality

Support data-driven business decisions

Manual analysis is inefficient at scale.
This project automates sentiment detection using machine learning.

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

Source: Amazon Reviews Dataset (Kaggle)

Format: CSV

Data Includes:

Review text

Product rating (1–5 stars)

Review metadata

Sentiment Mapping Logic:

4–5 stars → Positive

3 stars → Neutral

1–2 stars → Negative

Dataset not uploaded due to GitHub size limitations.

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

Python

Pandas & NumPy

Data manipulation

Feature engineering

NLTK

Stopword removal

Text preprocessing

Matplotlib & Seaborn

Data visualization

WordCloud

Text visualization

scikit-learn

TF-IDF Vectorizer

Logistic Regression

Model evaluation metrics

GitHub – Version control & documentation

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>
amazon-sentiment-analysis/
│
├── notebook/
│   └── Amazon_Sentiment_Analysis.ipynb
│
├── data/
│   └── (dataset not included)
│
├── requirements.txt
├── README.md
└── .gitignore

<h2><a class="anchor" id="data-cleaning--preparation"></a>Data Cleaning & Preparation</h2>

Data preprocessing steps performed:

Removal of null values

Lowercasing of text

Removal of special characters using Regex

Stopword removal using NLTK

Creation of sentiment labels from ratings

Feature extraction using TF-IDF (Unigrams + Bigrams)

<h2><a class="anchor" id="exploratory-data-analysis-eda"></a>Exploratory Data Analysis (EDA)</h2>

Key analysis performed:

Sentiment distribution

Rating distribution

Review length distribution

Review length comparison by sentiment

WordCloud visualization (Positive vs Negative reviews)

These insights helped understand class imbalance and textual behavior patterns.

<h2><a class="anchor" id="model-development"></a>Model Development</h2>

Problem Type: Multi-class Classification

Model Used:

Logistic Regression

Class balancing enabled

Stratified train-test split (80/20)

Feature Engineering:

TF-IDF Vectorization

max_features = 5000

ngram_range = (1,2)

<h2><a class="anchor" id="model-evaluation"></a>Model Evaluation</h2>

Evaluation metrics used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Multi-class ROC-AUC

Example Performance:

Accuracy: 0.89


(Replace with your actual model results.)

<h2><a class="anchor" id="key-insights--findings"></a>Key Insights & Findings</h2>

Positive reviews dominate the dataset

Negative reviews contain stronger emotional and descriptive language

Review length varies significantly by sentiment

Bigrams improved classification performance

Class weighting reduced imbalance impact

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

Clone the repository:

git clone https://github.com/Nityanand-Khule-24/amazon_product_sentiment_analysis.git


Install dependencies:

pip install -r requirements.txt


Open the notebook:

notebook/Amazon_Sentiment_Analysis.ipynb


Run all cells sequentially.

<h2><a class="anchor" id="future-improvements"></a>Future Improvements</h2>

Hyperparameter tuning using GridSearchCV

Compare with Naive Bayes and SVM

Implement transformer-based models (BERT)

Deploy model using Streamlit

Convert into REST API using Flask

<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

*** Nityanand Khule ***
Artificial Intelligence & Machine Learning Student
Savitribai Phule Pune University

📧 nityanandkhule24@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/nityanand-khule/