import numpy as np
import sqlalchemy as sqla
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from cleaning import clean_content
from scipy.sparse import save_npz

# conn_string = 'postgres://postgres@localhost/fakenews_data'
conn_string = 'postgresql://postgres@localhost/fake_news_corpus'
# conn_string = 'postgres://postgres@localhost/fake_news_250'
engine = sqla.create_engine(conn_string)
connection = engine.connect()

content_query = '''
SELECT cleaned_content, type
FROM all_articles
WHERE type IN ('fake', 'satire', 'bias', 'conspiracy', 'junksci', 'reliable', 'political', 'clickbait')
'''

data_frame = pd.read_sql(content_query, connection)
data_frame = data_frame.dropna(subset=['cleaned_content', 'type'])
data_frame = data_frame.sample(frac=0.1, random_state=0).reset_index(drop=True)
fake_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci']
data_frame['label'] = data_frame['type'].apply(lambda x: 1 if x in fake_labels else 0)

X = data_frame['cleaned_content'].to_numpy()
labels = data_frame['label'].to_numpy()


train_X, test_X, train_y, test_y = train_test_split(X, labels, test_size=0.2, random_state=0)

del X
del labels

vectorizer = TfidfVectorizer(min_df=10, max_df=0.99, dtype=np.float32)

train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

save_npz('train_X.npz', train_X)
np.save('train_y.npy', train_y)

save_npz('test_X.npz', test_X)
np.save('test_y.npy', test_y)

del train_X
del train_y
del test_X
del test_y


liar_test_data = pd.read_csv('test.tsv', delimiter='\t', header=None, index_col=None)

liar_fake_labels = ['pants-fire', 'false', 'barely-true']

liar_test_data['label'] = liar_test_data[1].apply(lambda x: 1 if x in liar_fake_labels else 0)

liar_test_data['cleaned'] = clean_content(liar_test_data[2])

liar_test_vectorized = vectorizer.transform(liar_test_data['cleaned'].to_numpy())

save_npz('liar_test_vectorized.npz', liar_test_vectorized)

np.save('liar_labels.npy', liar_test_data['label'].to_numpy())


kaggle_data = pd.read_json("test_set.json")
kaggle_data['cleaned'] = clean_content(kaggle_data['article'])

kaggle_vectorized = vectorizer.transform(kaggle_data['cleaned'].to_numpy())

save_npz('kaggle_vectorized.npz', kaggle_vectorized)
