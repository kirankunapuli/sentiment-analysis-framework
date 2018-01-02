# Natural Language Processing - Sentiment Analysis

# Importingthe libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords', download_dir = '/Users/kirankunapuli/anaconda3/lib/nltk_data')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

rf = RandomForestClassifier()
ada = AdaBoostClassifier()
lr = LogisticRegression()
nb = GaussianNB()
svm = LinearSVC()
mlp = MLPClassifier()

rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
ada_probas = ada.fit(X_train, y_train).predict_proba(X_test)
lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
svm_scores = svm.fit(X_train, y_train).decision_function(X_test)
mlp_probas = nb.fit(X_train, y_train).predict_proba(X_test)

probas_list = [rf_probas, ada_probas, lr_probas, nb_probas, svm_scores, mlp_probas]


clf_names = ['Random Forest', 'Ada Boost', 'Logistic Regression',
             'Gaussian Naive Bayes', 'Support Vector Machine', 'Neural Network']

skplt.metrics.plot_calibration_curve(y_test,probas_list,clf_names)
skplt.metrics.plot_confusion_matrix(y_test, ada.fit(X_train, y_train).predict(X_test), normalize=False)
