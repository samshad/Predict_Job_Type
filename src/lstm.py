import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dropout, GRU, Activation, \
    Embedding, Bidirectional, SpatialDropout1D, BatchNormalization, Conv1D, MaxPool1D
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

import PyPDF2
import text_cleaner as tc


def extract_text_from_pdf(file):
    f_reader = PyPDF2.PdfFileReader(open(file, 'rb'))
    page_count = f_reader.getNumPages()
    text = [f_reader.getPage(i).extractText() for i in range(page_count)]
    return str(text).replace("\\n", "")


df = pd.read_csv('Data/translated_file.csv')

MAX_NB_WORDS = 150000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
le = LabelEncoder()

df['category_num'] = le.fit_transform(df['category'])
idf = df[['category', 'category_num']].drop_duplicates()
categories = {}
for index, row in idf.iterrows():
    categories[row['category_num']] = row['category']

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, oov_token='OOV')
tokenizer.fit_on_texts(df['clean_description'].values)

'''word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))'''

X = tokenizer.texts_to_sequences(df['clean_description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['category_num'].values)

print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

job_model = Sequential()
job_model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

job_model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
job_model.add(LSTM(80, dropout=0.3, recurrent_dropout=0.3))

job_model.add(Dense(128, activation='relu'))
job_model.add(Dropout(0.3))

job_model.add(Dense(43, activation='softmax'))
job_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
batch_size = 64

history = job_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = job_model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

#job_model.save('job_model.h5')

resume_txt = tc.cleaner(extract_text_from_pdf('Data/Resumes/Md Samshad Rahman.pdf'))
seq = tokenizer.texts_to_sequences(resume_txt)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = job_model.predict(padded)
x = np.argmax(pred, axis=-1)
print(x)
print(pred)
print(categories[list(x)[0]])
