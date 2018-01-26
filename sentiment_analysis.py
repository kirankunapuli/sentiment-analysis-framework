# Natural Language Processing - Sentiment Analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
#import nltk
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

import scikitplot as skplt
#skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False, title = 'Naive Bayes')
#plt.show()

def plot_classifications():
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    rf = RandomForestClassifier()
    ada = AdaBoostClassifier()
    gb = GradientBoostingClassifier()
    lr = LogisticRegression()
    nb = GaussianNB()
    svm = LinearSVC()
    mlp = MLPClassifier()

    rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
    ada_probas = ada.fit(X_train, y_train).predict_proba(X_test)
    gb_probas = gb.fit(X_train, y_train).predict_proba(X_test)
    lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
    nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
    svm_scores = svm.fit(X_train, y_train).decision_function(X_test)
    mlp_probas = nb.fit(X_train, y_train).predict_proba(X_test)

    probas_list = [rf_probas, ada_probas, gb_probas, lr_probas, nb_probas, svm_scores, mlp_probas]

    clf_names = ['Random Forest', 'Ada Boost', 'Gradient Boosting', 'Logistic Regression',
                 'Gaussian Naive Bayes', 'Support Vector Machine', 'Neural Network']

    import scikitplot as skplt
    
    print('\n')
    print('Calibration Curve:\n')
    skplt.metrics.plot_calibration_curve(y_test,probas_list,clf_names)
    plt.show()
    
    # Plotting classification reports
    print('\n')
    print("Random Forest ROC-AUC score: %.3f" % roc_auc_score(y_test, rf_probas[:, 1]))
    print('\n')
    print('Classification report for Random Forest\n',
          classification_report(y_test, rf.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, rf.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='Random Forest')
    plt.show()

    print('\n')
    print("AdaBoost ROC-AUC score: %.3f" % roc_auc_score(y_test, ada_probas[:, 1]))
    print('\n')
    print('Classification report for AdaBoost\n',
          classification_report(y_test, ada.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, ada.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='AdaBoost')
    plt.show()

    print('\n')
    print("Gradient Boosting Classifier ROC-AUC score: %.3f" % roc_auc_score(y_test, ada_probas[:, 1]))
    print('\n')
    print('Classification report for Gradient Boosting Classifier\n',
          classification_report(y_test, gb.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, gb.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='Gradient Boosting')
    plt.show()

    print('\n')
    print("Logistic Regression ROC-AUC score: %.3f" % roc_auc_score(y_test, lr_probas[:, 1]))
    print('\n')
    print('Classification report for Logistic Regression\n',
          classification_report(y_test, lr.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, lr.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='Logistic Regression')
    plt.show()

    print('\n')
    print("Naive Bayes ROC-AUC score: %.3f" % roc_auc_score(y_test, nb_probas[:, 1]))
    print('\n')
    print('Classification report for Naive Bayes\n',
          classification_report(y_test, nb.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, nb.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='Naive Bayes')
    plt.show()

    print('\n')
    # print("SVClassifier ROC-AUC score: %.3f" % roc_auc_score(y_test, svm_scores[:, 1]))
    # print('\n')
    print('Classification report for SVClassifier\n',
          classification_report(y_test, svm.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, svm.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='SVClassifier')
    plt.show()

    print('\n')
    print("Neural Network ROC-AUC score: %.3f" % roc_auc_score(y_test, mlp_probas[:, 1]))
    print('\n')
    print('Classification report for Neural Network\n',
          classification_report(y_test, mlp.fit(X_train, y_train).predict(X_test)))
    skplt.metrics.plot_confusion_matrix(y_test, mlp.fit(X_train, y_train).predict(X_test), normalize=False,
                                        title='Neural Network')
    plt.show()

# Display plots of all classifications
plot_classifications()
