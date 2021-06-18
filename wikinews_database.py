import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from cleaning import clean_content, make_list_atomic
import sqlalchemy as sqla


concurrent_load = False

conn_string = 'postgresql://postgres@localhost/fake_news_250'

engine = sqla.create_engine(conn_string)

connection = engine.connect()


table_create_sql = '''
CREATE TABLE IF NOT EXISTS wikinews_articles (
    id INT PRIMARY KEY,
    url TEXT,
    title TEXT,
    simple_date TEXT,
    full_date TEXT,
    cleaned_content TEXT,
    scraped_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wikinews_sources (
    source_text TEXT,
    source_link TEXT,
    article_id INT,
    PRIMARY KEY (source_text, source_link, article_id),
    FOREIGN KEY (article_id) REFERENCES wikinews_articles(id)
);

CREATE TABLE IF NOT EXISTS wikinews_categories (
    category_name TEXT,
    article_id INT,
    PRIMARY KEY (category_name, article_id),
    FOREIGN KEY (article_id) REFERENCES wikinews_articles(id)
);
'''

connection.execute(table_create_sql)

array_pattern = re.compile(r'\'(.*?[^\\])\'')

def load(source_data):
    source_data['cleaned_content'] = clean_content(source_data['content'], not concurrent_load)
    source_data = source_data.dropna(subset=['cleaned_content'])

    database_articles = source_data[['id', 'url', 'title', 'simple_date', 'full_date', 'cleaned_content',
                                     'scraped_at', 'updated_at']]

    database_categories = make_list_atomic(source_data, 'id', 'categories', array_pattern).drop_duplicates().dropna()
    database_categories = database_categories.rename(columns={'categorie_name': 'category_name', 'article_index': 'article_id'})

    database_source_texts = make_list_atomic(source_data, 'id', 'source_texts', array_pattern)
    database_source_links = make_list_atomic(source_data, 'id', 'source_links', array_pattern)

    database_sources = database_source_texts.merge(database_source_links, on=None, left_index=True, right_index=True)[['article_index_x', 'source_text_name', 'source_link_name']]
    database_sources = database_sources.rename(columns={'article_index_x': 'article_id', 'source_text_name': 'source_text', 'source_link_name': 'source_link'})
    database_sources = database_sources.drop_duplicates().dropna()

    database_articles.to_sql('wikinews_articles', connection, if_exists='append', index=False)
    database_categories.to_sql('wikinews_categories', connection, if_exists='append', index=False)
    database_sources.to_sql('wikinews_sources', connection, if_exists='append', index=False)


source_data = pd.read_csv('wikinews_data.csv')

load(source_data)

connection.close()
