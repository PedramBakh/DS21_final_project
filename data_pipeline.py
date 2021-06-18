import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from cleaning import clean_content, make_list_atomic
import sqlalchemy as sqla
import sys
import csv

nrows = 250000
chunksize = 500

concurrent_load = False

conn_string = 'postgresql://postgres@localhost/fake_news_250'

engine = sqla.create_engine(conn_string)

connection = engine.connect()


table_create_sql = '''
CREATE TABLE IF NOT EXISTS Articles (
    index INT PRIMARY KEY,
    id INT,
    title TEXT,
    url TEXT,
    domain TEXT,
    type TEXT,
    cleaned_content TEXT,
    scraped_at TIMESTAMP,
    inserted_at TIMESTAMP,
    updated_at TIMESTAMP,
    meta_description TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS Authors (
    author_name TEXT,
    article_index INT,
    PRIMARY KEY (author_name, article_index),
    FOREIGN KEY (article_index) REFERENCES Articles(index)
);

CREATE TABLE IF NOT EXISTS Keywords (
    keyword_name TEXT,
    article_index INT,
    PRIMARY KEY (keyword_name, article_index),
    FOREIGN KEY (article_index) REFERENCES Articles(index)
);

CREATE TABLE IF NOT EXISTS Meta_Keywords (
    meta_keyword_name TEXT,
    article_index INT,
    PRIMARY KEY (meta_keyword_name, article_index),
    FOREIGN KEY (article_index) REFERENCES Articles(index)
);

CREATE TABLE IF NOT EXISTS Tags (
    tag_name TEXT,
    article_index INT,
    PRIMARY KEY (tag_name, article_index),
    FOREIGN KEY (article_index) REFERENCES Articles(index)
);
'''

trigger_create_sql = '''
CREATE OR REPLACE FUNCTION pass() RETURNS trigger AS $$
    BEGIN
        RETURN NULL;
    END;
$$ LANGUAGE plpgsql ;

CREATE TRIGGER InsertedNotNull
    BEFORE INSERT ON Authors
    FOR EACH ROW
    WHEN ( NEW.author_name IS NULL )
    EXECUTE PROCEDURE pass();

CREATE TRIGGER InsertedNotNull
    BEFORE INSERT ON Keywords
    FOR EACH ROW
    WHEN ( NEW.keyword_name IS NULL )
    EXECUTE PROCEDURE pass();

CREATE TRIGGER InsertedNotNull
    BEFORE INSERT ON Meta_Keywords
    FOR EACH ROW
    WHEN ( NEW.meta_keyword_name IS NULL )
    EXECUTE PROCEDURE pass();

CREATE TRIGGER InsertedNotNull
    BEFORE INSERT ON Tags
    FOR EACH ROW
    WHEN ( NEW.tag_name IS NULL )
    EXECUTE PROCEDURE pass();
'''

connection.execute(table_create_sql)
connection.execute(trigger_create_sql)


array_pattern = re.compile(r'[^, \[\]\']+(?: [^, \[\]\']+)*')
list_pattern = re.compile(r'[^, ]+(?: [^, ]+)*')

id_pattern = re.compile(r'^\d+$')


def load(source_data, current_index):
    source_data = source_data[source_data['id'].str.contains(id_pattern)].copy(deep=True)

    source_data['cleaned_content'] = clean_content(source_data['content'], not concurrent_load)

    source_data = source_data.dropna(subset=['cleaned_content', 'id', 'type'])

    source_data['index'] = source_data.index + current_index

    database_authors = make_list_atomic(source_data, 'index', 'authors', list_pattern).drop_duplicates()
    database_keywords = make_list_atomic(source_data, 'index', 'keywords', list_pattern).drop_duplicates()
    database_meta_keywords = make_list_atomic(source_data, 'index', 'meta_keywords', array_pattern).drop_duplicates()
    database_tags = make_list_atomic(source_data, 'index', 'tags', list_pattern).drop_duplicates()

    database_articles = source_data[['index', 'id', 'domain', 'type', 'url', 'cleaned_content',
                                     'scraped_at', 'inserted_at', 'updated_at', 'title', 'meta_description', 'summary']]

    database_articles.to_sql('articles', connection, if_exists='append', index=False, method='multi')
    database_authors.to_sql('authors', connection, if_exists='append', index=False, method='multi')
    database_keywords.to_sql('keywords', connection, if_exists='append', index=False, method='multi')
    database_meta_keywords.to_sql('meta_keywords', connection, if_exists='append', index=False, method='multi')
    database_tags.to_sql('tags', connection, if_exists='append', index=False, method='multi')

    if concurrent_load:
        print(f'Startindex {current_index} done', flush=True)


executor = ProcessPoolExecutor(max_workers=8)

csv_iterator = pd.read_csv("250t-raw.csv", index_col=None, engine='python', chunksize=chunksize, dtype={"id": "string", "domain": "string", "type": "string", "url": "string", "content": "string", "scraped_at": "string", "inserted_at": "string", "updated_at": "string", "title": "string", "authors": "string", "keywords": "string", "meta_keywords": "string", "meta_description": "string", "tags": "string", "summary": "string", "source": "string"})

print('Starting')

# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


if concurrent_load:
    executor.map(load, csv_iterator, range(0, nrows, chunksize))
else:
    with tqdm(total=nrows, unit='rows') as pbar:
        for current_index, source_data in enumerate(csv_iterator):
            load(source_data, current_index * chunksize)
            pbar.update(chunksize)

connection.close()
