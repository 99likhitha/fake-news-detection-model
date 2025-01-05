Fake News Prediction Model
This project is designed to predict whether a given news article is real or fake using machine learning techniques. The model uses natural language processing (NLP) methods to process text data and train a classification model. The primary goal is to identify fake news based on the textual content of the article.

Technologies Used
Python: The main programming language used for this project.
Libraries:
nltk: For text processing, including stopword removal and stemming.
scikit-learn: For machine learning model building and vectorization.
pandas: For data manipulation.
numpy: For numerical operations.
streamlit: For the web-based interactive interface.
Dataset
The dataset used in this project is from Kaggle, which contains labeled news articles classified as either real or fake. The dataset includes various features, but the primary focus is on the title and author columns.

Model
Text Preprocessing: The text data is preprocessed by:

Removing special characters and non-alphabetical characters.
Converting the text to lowercase.
Splitting the text into words.
Removing stopwords (common words like 'the', 'is', 'and', etc.).
Applying stemming to reduce words to their base form.
Vectorization: The text is converted into a numerical format using TF-IDF vectorization to capture the importance of words in the context of the entire dataset.

Logistic Regression: A logistic regression classifier is used to predict whether a given news article is real or fake. The model is trained on 80% of the data, and the remaining 20% is used for testing.

Streamlit Application
The project also includes a Streamlit web application that allows users to interact with the model and make predictions on news articles. Users can input text into the web interface, and the model will predict if the news is fake or real.
