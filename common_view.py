import sqlalchemy as sqla

conn_string = 'postgres://postgres@localhost/fake_news_250'

engine = sqla.create_engine(conn_string)

connection = engine.connect()

create_articles_view_sql = '''
CREATE OR REPLACE VIEW all_articles (
    fakenewscorpus_index,
    wikinews_id,
    title,
    type,
    cleaned_content,
    url,
    scraped_at,
    updated_at
)
AS
(
    SELECT NULL::INT, id, title, 'real', cleaned_content, url, scraped_at, updated_at
    FROM wikinews_articles
)
UNION
(
    SELECT index, NULL::INT, title, type, cleaned_content, url, scraped_at, updated_at
    FROM articles
)
'''

connection.execute(create_articles_view_sql)

create_keyword_view_sql = '''
CREATE OR REPLACE VIEW all_keywords (
    fakenewscorpus_index,
    wikinews_id,
    keyword_name
)
AS
(
    SELECT article_index, NULL::INT, keyword_name
    FROM keywords
)
UNION
(
    SELECT article_index, NULL::INT, meta_keyword_name
    FROM meta_keywords
)
UNION
(
    SELECT article_index, NULL::INT, tag_name
    FROM tags
)
UNION
(
    SELECT NULL::INT, article_id, category_name
    FROM wikinews_categories
)
'''

connection.execute(create_keyword_view_sql)
