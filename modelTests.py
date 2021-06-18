import numpy as np
import sqlalchemy as sqla
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
# from sklearn.metrics import
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, LogisticRegression
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import save_npz, load_npz, csr_matrix
from deepLearnModel import LinearSequentialClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.preprocessing import StandardScaler
from cleaning import clean_content

# conn_string = 'postgres://postgres@localhost/fakenews_data'
conn_string = 'postgres://postgres@localhost/fake_news_corpus'
# conn_string = 'postgresql://postgres@localhost/fake_news_corpus'
# conn_string = 'postgres://postgres@localhost/fake_news_250'

engine = sqla.create_engine(conn_string)

connection = engine.connect()

content_query = '''
SELECT cleaned_content, type
FROM all_articles
WHERE type IN ('fake', 'satire', 'bias', 'conspiracy', 'junksci', 'reliable', 'political', 'clickbait')
'''

database_iterator = pd.read_sql(content_query, connection, chunksize=250000)

data_frame = next(database_iterator)

data_frame = data_frame.dropna(subset=['cleaned_content', 'type'])

fake_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci']

data_frame['label'] = data_frame['type'].apply(lambda x: 1 if x in fake_labels else 0)

labels = data_frame['label'].to_numpy()  # .reshape(-1, 1)

X = data_frame['cleaned_content'].to_numpy()


vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.99, dtype=np.float32)
# vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9, dtype=np.float32)
# vectorizer = TfidfVectorizer(dtype=np.float32)
# vectorizer = HashingVectorizer()
# vectorizer = CountVectorizer(dtype=np.int32)


train_X, test_X, train_y, test_y = train_test_split(X, labels, test_size=0.4, random_state=0)

train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

kaggle_data = pd.read_json("test_set.json")
kaggle_data['cleaned'] = clean_content(kaggle_data['article'])

kaggle_vectorized = vectorizer.transform(kaggle_data['cleaned'].to_numpy())

# save_npz('train_X.npz', train_X)
# save_npz('test_X.npz', test_X)
# np.save('train_y.npy',train_y)
# np.save('test_y.npy', test_y)

# train_X = load_npz('train_X.npz')
# test_X = load_npz('test_X.npz')
# train_y = np.load('train_y.npy')
# test_y = np.load('test_y.npy')


test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=0)


print(f'Train data X shape: {train_X.shape}')
print(f'Train data labels distribution: {np.bincount(train_y)}')
print(f'Validation data X shape: {val_X.shape}')
print(f'Validation data labels distribution: {np.bincount(val_y)}')
print(f'Test data X shape: {test_X.shape}')
print(f'Test data labels distribution: {np.bincount(test_y)}')


models = [
    # LinearSVC(),
    # SGDClassifier(loss='hinge', n_jobs=-1),
    # KNeighborsClassifier(n_jobs=-1),
    # MultinomialNB(),
    # LogisticRegression(),
    # SGDClassifier(loss='log', n_jobs=-1),
    # RidgeClassifier(max_iter=100),
    # DecisionTreeClassifier(),
    # ComplementNB(),
    # Perceptron(),
    PassiveAggressiveClassifier(),
    # SGDClassifier(n_jobs=-1),
    # MLPClassifier(hidden_layer_sizes=(10,)),
    # RandomForestClassifier(n_jobs=-1),
    # AdaBoostClassifier(),
    LinearSequentialClassifier([1000, 1000, 500, 2])
]


def test_model(input_model, vectorizer=None):
    if vectorizer:
        model = Pipeline(
            [('vectorizer', vectorizer()),
            ('model', input_model)]
        )
    else:
        model = input_model

    model.fit(train_X, train_y)
    score = model.score(val_X, val_y)

    model_name = str(input_model).split('(')[0]

    print(f"Model: {model_name}, accuracy: {score}")

    kaggle_pred = model.predict(kaggle_vectorized)

    print(f'Kaggle distribution: {np.bincount(kaggle_pred)}', flush=True)

    kaggle_data = pd.read_json("test_set.json")

    kaggle_data['label'] = ['FAKE' if x == 1 else 'REAL' for x in kaggle_pred]

    kaggle_data[['id', 'label']].to_csv(f'kaggle_predictions_{model_name + str(score)}.csv', index=False)

    return score


def test_all_models(models, vectorizer=None):
    xs = np.arange(len(models))
    scores = np.zeros(len(xs))
    fig, ax = plt.subplots()

    model_names = [str(model).split('(')[0] for model in models]

    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     scores = executor.map(test_model, models)
    #
    #     for i, score in enumerate(scores):
    #         plt.bar(xs[i], score)

    with tqdm(total=len(models), unit='models') as pbar:
        for i in range(len(models)):
            scores[i] = test_model(models[i], vectorizer)
            plt.bar(xs[i], scores[i])
            pbar.update(1)

    fig.autofmt_xdate()
    plt.xticks(xs, model_names)
    plt.show()


# test_model(models[4])
test_all_models(models)
