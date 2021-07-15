import PyPDF2
import text_cleaner as tc
import pickle
import pandas as pd


def extract_text_from_pdf(file):
    f_reader = PyPDF2.PdfFileReader(open(file, 'rb'))
    page_count = f_reader.getNumPages()
    text = [f_reader.getPage(i).extractText() for i in range(page_count)]
    return str(text).replace("\\n", "")


def get_job_class(doc):
    with open('Model/vectorizer.pkl', 'rb') as pickle_file:
        word_vectorizer = pickle.load(pickle_file)
    with open('Model/knn.pkl', 'rb') as pickle_file:
        knn = pickle.load(pickle_file)
    with open('Model/label_encoder.pkl', 'rb') as pickle_file:
        le = pickle.load(pickle_file)

    doc = tc.cleaner(doc)
    pred_txt = word_vectorizer.transform([doc])
    prediction = knn.predict(pred_txt)
    #print(prediction)
    return le.inverse_transform(prediction)


"""resume_txt = tc.cleaner(extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf'))
pred_txt = word_vectorizer.transform([resume_txt])

print(pred_txt.shape)
prediction = knn.predict(pred_txt)
print(prediction)
print(le.inverse_transform(prediction))

df = pd.read_csv('Data/Resumes/Archive/ResumeDataSet_1.csv')
print(df['category'].value_counts())
# tf = df[df['category'] == 'HR']
# tf = df[df['category'] == 'Java Developer']
tf = df[df['category'] == 'Mechanical Engineer']
# tf = df[df['category'] == 'Business Analyst']
print(tf['category'].value_counts())

for index, row in tf.iterrows():
    resume_txt = tc.cleaner(row['resume'])
    pred_txt = word_vectorizer.transform([resume_txt])
    prediction = knn.predict(pred_txt)
    print(prediction)
    print(le.inverse_transform(prediction))
"""
