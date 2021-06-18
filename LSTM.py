import keras
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from keras import layers
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from cleaning import clean_content

conn_string = 'postgresql://postgres@localhost/fake_news_corpus'
engine = sqla.create_engine(conn_string)
connection = engine.connect()

content_query = '''
SELECT cleaned_content, type
FROM all_articles
WHERE type IN ('fake', 'satire', 'bias', 'conspiracy', 'junksci', 'reliable', 'political', 'clickbait')
'''
chunksize = 150000
database_iterator = pd.read_sql(content_query, connection, chunksize=chunksize)
data_frame = next(database_iterator)
data_frame = data_frame.dropna(subset=['cleaned_content', 'type'])
data_frame = data_frame.sample(frac=0.1).reset_index(drop=True)
fake_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci']
data_frame['label'] = data_frame['type'].apply(lambda x: 1 if x in fake_labels else 0)
labels = data_frame['label'].to_numpy()
X = data_frame['cleaned_content'].to_numpy()

kaggle_data = pd.read_json("test_set.json")
kaggle_data['cleaned'] = clean_content(kaggle_data['article'])
kaggle_X = kaggle_data['cleaned'].to_numpy()

train_X, test_X, train_y, test_y = train_test_split(X, labels, test_size=0.2, random_state=0)
test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)

print(f'Train data X shape: {train_X.shape}')
print(f'Train data labels distribution: {np.bincount(train_y)}')
print(f'Test data X shape: {test_X.shape}')
print(f'Test data labels distribution: {np.bincount(test_y)}')

conn_string = 'postgresql://postgres@localhost/fake_news_corpus'
engine = sqla.create_engine(conn_string)
connection = engine.connect()
content_query = '''
SELECT cleaned_content, type
FROM all_articles
WHERE type IN ('fake', 'satire', 'bias', 'conspiracy', 'junksci', 'reliable', 'political', 'clickbait')
'''

class LSTM_model:
    def __init__(self, max_words=2000, max_len=300, loss_fn='binary_crossentropy',
                 optimizer='Adam', metrics=['accuracy'],
                 batch_size=8, epochs=25):
        self.max_words = max_words
        self.max_len = max_len
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.tokenizer = Tokenizer(num_words=self.max_words)


    def RNN(self):
        model = keras.Sequential()
        model.add(layers.Embedding(self.max_words, 512, mask_zero=True))
        model.add(layers.LSTM(128))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8))
        model.add(layers.Activation('tanh'))
        model.add(layers.Dense(4))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(1))
        model.add(layers.Activation('sigmoid'))

        return model

    def fit(self, X, y):
        self.tokenizer.fit_on_texts(X)
        seq = self.tokenizer.texts_to_sequences(X)
        seq_matrix = sequence.pad_sequences(seq, maxlen=self.max_len)

        self.model = self.RNN()
        self.model.summary()
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics)
        self.model.fit(seq_matrix, y, batch_size=self.batch_size, epochs=self.epochs,
                  validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)],
                       workers=2, shuffle=True)

    def score(self, X, y):
        seq = self.tokenizer.texts_to_sequences(X)
        seq_matrix = sequence.pad_sequences(seq, maxlen=self.max_len)
        accuracy = self.model.evaluate(seq_matrix, y, workers=2, batch_size=10)

        return accuracy[1]

    def predict(self, X):
        seq = self.tokenizer.texts_to_sequences(X)
        seq_matrix = sequence.pad_sequences(seq, maxlen=self.max_len)
        preds = self.model.predict(seq_matrix, workers=2)
        print(preds)
        preds = [1 if x > 0.5 else 0 for x in preds]

        print(np.bincount(preds))

        return preds


model = LSTM_model()
model.fit(train_X, train_y)
model.score(test_X, test_y)
model.predict(test_X)

kaggle_preds = model.predict(kaggle_X)
kaggle_data_temp = kaggle_data.copy(deep=True)
kaggle_data_temp['label'] = ['FAKE' if x == 1 else 'REAL' for x in kaggle_preds]
kaggle_data_temp[['id', 'label']].to_csv(f'kaggle_predictions_final_LSTM.csv', index=False)
