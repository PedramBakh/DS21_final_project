import pandas as pd
from cleaning import clean_content

liar_test_data = pd.read_csv('test.tsv', delimiter='\t', header=None, index_col=None)

liar_fake_labels = ['pants-fire', 'false', 'barely-true']

liar_test_data['label'] = liar_test_data[1].apply(lambda x: 1 if x in liar_fake_labels else 0)

liar_test_data['cleaned'] = clean_content(liar_test_data[2])


kaggle_data = pd.read_json("test_set.json")

kaggle_data['cleaned'] = clean_content(kaggle_data['article'])

print(liar_test_data['cleaned'].str.findall(r'\w+').apply(lambda x: len(x)).mean())

print(kaggle_data['cleaned'].str.findall(r'\w+').apply(lambda x: len(x)).mean())
