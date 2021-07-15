import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import PyPDF2
import text_cleaner as tc
import pickle


def extract_text_from_pdf(file):
    f_reader = PyPDF2.PdfFileReader(open(file, 'rb'))
    page_count = f_reader.getNumPages()
    text = [f_reader.getPage(i).extractText() for i in range(page_count)]
    return str(text).replace("\\n", "")


df = pd.read_csv('Data/train_jobs.csv')

le = LabelEncoder()
df['category_num'] = le.fit_transform(df['category'])

requiredText = df['clean_description'].values
requiredTarget = df['category_num'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=150000)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print("Feature completed .....")

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42, test_size=0.1)
print(X_train.shape)
print(X_test.shape)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# prediction = clf.predict(X_test)
# print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))

resume_txt = tc.cleaner(extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf'))
# word_vectorizer.fit([resume_txt])
pred_txt = word_vectorizer.transform([resume_txt])

print(pred_txt.shape)
prediction = clf.predict(pred_txt)
print(le.inverse_transform(prediction))
print("========================================")

clf2 = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
prediction = clf2.predict(pred_txt)
print(le.inverse_transform(prediction))

with open('Model/vectorizer.pkl', 'wb') as f:
    pickle.dump(word_vectorizer, f)

with open('Model/knn.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('Model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
