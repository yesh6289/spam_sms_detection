#program to detect spam sms

#importing the required libraries

import re
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
np.random.seed(20)

spam_df = pd.read_csv("spam.csv", encoding='latin-1', usecols=['v1', 'v2'])
spam_df.columns = ["label", "text"]
spam_df.head(5)

"""Helper Functions
plot_confusion_matrix: This function generates and visualizes a confusion matrix for the given model's predictions on the input data, including accuracy and normalized values.
build_model_pipeline: It creates a text classification pipeline using TF-IDF vectorization and the specified model, streamlining the model-building process.
plot_roc: This function plots the Receiver Operating Characteristic (ROC) curve for the model's predictions, providing insights into its performance and area under the curve (AUC) score.
evaluate_model: Given a model and input data, it evaluates and returns a DataFrame with precision, recall, and F1-score metrics for each class in the classification task, facilitating model assessment and comparison.**bold text** **bold text**
"""

def plot_confusion_matrix(model, X, y, ax, **kwargs):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    cm_norm = cm / cm.sum(1).reshape((-1, 1))
    labels = [f"{v}\n({pct:.2%})" for v, pct in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.array(labels).reshape(cm.shape)

    sns.heatmap(cm, annot=labels, fmt="",
                ax=ax, yticklabels=le.classes_,
                xticklabels=le.classes_, cbar=False,
               cmap=["#1f1e1a", "#b09302"])
    ax.set_title(f"{kwargs.get('name').upper()} - {acc:.2%}", size=10.5, alpha=0.65)


def build_model_pipeline(model, name):
    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(min_df=2, stop_words=list(ENGLISH_STOP_WORDS))),
        (name, model)
    ])
    return pipeline


def plot_roc(model, X, y, ax, **kwargs):
    pred_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred_prob)
    fpr, tpr, threshold = roc_curve(y, pred_prob)

    ax.plot(fpr, tpr, label=f'{kwargs.get("name")} - AUC = {auc:.2%}')
    straight_line = np.linspace(0, 1, len(fpr))
    ax.plot(straight_line, straight_line)
    ax.fill_between(fpr, fpr, tpr, alpha=0.1)
    ax.legend(loc=4, frameon=True, edgecolor="gray")
    ax.set_title("ROC Curve", size=11)
    ax.grid(visible=False, axis="y")
    ax.set(ylabel="tpr", xlabel="fpr")


def evaluate_model(model, X, y, name):
    pred = model.predict(X)
    prf = precision_recall_fscore_support(y, pred)[:-1]
    metrics = "precision_recall_fscore".split("_")
    arrays = [["", name, ""], metrics]
    index = pd.MultiIndex.from_arrays(arrays, names=('model', 'metric'))
    df = pd.DataFrame(prf, columns=le.classes_, index=index)

    return df

"""EDA"""

fig = plt.figure(figsize=(5, 3))
sns.countplot(data=spam_df, x='label', palette=["#b09302", "#000000"])
plt.title("Counts of Labels")
plt.show()

spam = spam_df[spam_df.label=='spam']
ham = spam_df[spam_df.label=='ham'].sample(spam.shape[0])

spam_df = pd.concat([spam, ham], axis=0)

spam_df['length'] = spam_df.text.apply(len)

fig = plt.figure(figsize=(5, 3))
sns.kdeplot(data=spam_df, x='length', color = '#b09302')
plt.title("Distribution of texts length")
plt.show()

"""Preprocessing **Text**"""

def process_text(message):
    text_split = message.lower().split()
    texts = [word for word in text_split if word not in string.punctuation]
    texts = [re.sub("[^a-zA-Z]", " ", word).strip() for word in texts]
    texts = " ".join(texts)
    texts = [word for word in texts.split() if word not in ENGLISH_STOP_WORDS and len(word)>1]

    return " ".join(texts)

process_text("Todays Voda numbers ending 1225 are selected to receive a å£50award. \
            If you have a match please call 08712300220 quoting claim code 3100 standard rates app")

spam_df['text'] = spam_df['text'].apply(process_text)

spam_df.sample(5)

le = LabelEncoder()

spam_df.reset_index(drop=True, inplace=True)
x = spam_df.text
y = le.fit_transform(spam_df.label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""**Building Models Pipeline**"""

models = {
    "nb": build_model_pipeline(MultinomialNB(), "nb").fit(x_train, y_train),
    "knn": build_model_pipeline(KNeighborsClassifier(n_neighbors=70), "knn").fit(x_train, y_train),
    "lr": build_model_pipeline(LogisticRegression(solver="liblinear"), "lr").fit(x_train, y_train),
    "rfc": build_model_pipeline(RandomForestClassifier(max_depth=10), "rfc").fit(x_train, y_train)
}

"""**Evaluating Models**"""

fig = plt.figure(figsize=(8, 7))

for i, (model_name, model) in enumerate(models.items(), start=1):
    ax = fig.add_subplot(2, 2, i, mouseover=True)
    plot_confusion_matrix(model, x_test, y_test, ax, name = model_name)
fig.supxlabel("Predicted class")
fig.supylabel("Actual class", x=0.02)
fig.suptitle("Confusion Matrix", size=16)
plt.tight_layout(pad=1.4)
fig.patch.set_alpha(0)
plt.show()

ax = plt.axes()
plot_roc(models['nb'], x_test, y_test, ax, name='nb')
plot_roc(models['knn'], x_test, y_test, ax, name='knn')
plot_roc(models['lr'], x_test, y_test, ax, name='lr')
plot_roc(models["rfc"], x_test, y_test, ax, name='rfc')
ax.set_alpha(0)
plt.show()

"""**Interpretation of AUC**

Naive Bayes: AUC at 0.9947 reflects an outstanding ability to separate
positive and negative instances, showcasing the model's high accuracy in classification tasks.

KNN: Despite a slightly lower AUC at 0.9246, KNN displays good discriminatory prowess, striking a balance between true positives and false positives.

Logistic Regression: The exceptionally high AUC of 0.9973 underscores the model's remarkable accuracy, indicating superior performance in correctly classifying instances.

Random Forest: With an AUC of 0.9947, Random Forest showcases robust discriminatory performance, akin to Naive Bayes.



"""

results_df = None

for i, (model_name, model) in enumerate(models.items()):
    model_results = evaluate_model(
        models[model_name], x_test, y_test, name=model_name.upper())
    if i == 0:
        results_df = model_results.copy()
    else:
        results_df = pd.concat((results_df, model_results))

results_df
