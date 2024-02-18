# spam_sms_detection
detecting spam sms using knn model 

Creating a spam SMS detection program in Python using machine learning involves several steps, including data collection, preprocessing, feature extraction, model training, and evaluation. Here's a detailed description of each step along with a basic implementation using Python and common machine learning libraries such as scikit-learn.

Step 1: Data Collection
The first step is to collect a dataset containing SMS messages labeled as spam or not spam (ham). A popular dataset used for this purpose is the SMS Spam Collection Dataset, which is publicly available and contains a set of labeled messages.

Step 2: Data Preprocessing
Data preprocessing involves cleaning and preparing the text data for modeling. This may include:

Removing special characters and numbers.
Converting all messages to lowercase to ensure consistency.
Tokenizing the text into individual words.
Removing stopwords (common words that are unlikely to be useful for prediction).
Applying stemming or lemmatization to reduce words to their base form.
Step 3: Feature Extraction
After preprocessing, the next step is to convert the text data into numerical format that can be used by machine learning models. A common approach is to use the Bag of Words model or TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into a vector form.

Step 4: Model Training
Once we have our features, we can train a machine learning model to classify messages as spam or ham. Commonly used algorithms for text classification include Naive Bayes, Support Vector Machine (SVM), and Logistic Regression.

Step 5: Evaluation
After training the model, it's important to evaluate its performance using metrics such as accuracy, precision, recall, and F1 score. This can be done by splitting the dataset into training and testing sets or using cross-validation.

This is a basic implementation. The performance can be improved by experimenting with different preprocessing steps, feature extraction techniques, and machine learning models. Additionally, consider using more advanced techniques like word embeddings or deep learning models for better accuracy.


<img width="219" alt="Screenshot 2024-02-18 230625" src="https://github.com/yesh6289/spam_sms_detection/assets/150354961/5b2d8da7-2263-4517-af0f-b28d3d5f9301">
<img width="412" alt="Screenshot 2024-02-18 230609" src="https://github.com/yesh6289/spam_sms_detection/assets/150354961/1934e55a-0291-4432-b289-d67917a6d722">
<img width="580" alt="Screenshot 2024-02-18 230555" src="https://github.com/yesh6289/spam_sms_detection/assets/150354961/cfe5d450-dfaa-4aa6-84f3-933b56d4459c">
<img width="345" alt="Screenshot 2024-02-18 230447" src="https://github.com/yesh6289/spam_sms_detection/assets/150354961/6c9eb5d5-0e74-461e-8657-4a88a179439e">
<img width="359" alt="Screenshot 2024-02-18 230424" src="https://github.com/yesh6289/spam_sms_detection/assets/150354961/e7b14903-7c21-44f9-820c-41d575301c31">
