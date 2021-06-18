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
from scipy.sparse import save_npz, load_npz
from sklearn.metrics import confusion_matrix, classification_report
import pickle


train_X = load_npz('train_X.npz')
train_y = np.load('train_y.npy')

print(f'Train data X shape: {train_X.shape}', flush=True)
print(f'Train data labels distribution: {np.bincount(train_y)}', flush=True)

models = [
    # SGDClassifier(loss='hinge', n_jobs=-1),
    # MultinomialNB(),
    # SGDClassifier(loss='log', n_jobs=-1),
    LinearSequentialClassifier([512, 256, 128, 64, 32, 2], learning_rate=1e-3, weight_decay=0, epochs=2, train_batch_size=10),
]

for i, model in enumerate(models):
    model.fit(train_X, train_y)

    with open(f'model{i}.pkl', 'wb') as file:
        pickle.dump(model, file)


del train_X
del train_y


test_X = load_npz('test_X.npz')
test_y = np.load('test_y.npy')

print(f'Test data X shape: {test_X.shape}', flush=True)
print(f'Test data labels distribution: {np.bincount(test_y)}', flush=True)


liar_test_vectorized = load_npz('liar_test_vectorized.npz')
liar_test_labels = np.load('liar_labels.npy')

kaggle_vectorized = load_npz('kaggle_vectorized.npz')

kaggle_data = pd.read_json("test_set.json")


def report_model(model, test_X, test_y):
    preds = model.predict(test_X)

    print(confusion_matrix(test_y, preds), flush=True)

    print(classification_report(test_y, preds), flush=True)

    return (preds == test_y).sum() / len(test_y)


for model in models:
    model_name = str(model).split('(')[0]

    print(f'\nTesting {model_name} -------------------', flush=True)

    print('Test data report:', flush=True)
    test_score = report_model(model, test_X, test_y)

    print('LIAR data report:', flush=True)
    liar_score = report_model(model, liar_test_vectorized, liar_test_labels)

    print(f"Model: {model_name}:, \n-->test data accuracy: {test_score}, \n-->liar data accuracy: {liar_score}", flush=True)

    kaggle_preds = model.predict(kaggle_vectorized)

    kaggle_preds_df = kaggle_data[['id']]

    kaggle_preds_df['label'] = ['FAKE' if x == 1 else 'REAL' for x in kaggle_preds]

    kaggle_preds_df.to_csv(f'kaggle_predictions_{model_name + str(test_score)}.csv', index=False)
