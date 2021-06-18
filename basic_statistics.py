import sqlalchemy as sqla
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median

conn_string = 'postgresql://postgres@localhost/fake_news_corpus'

engine = sqla.create_engine(conn_string)

connection = engine.connect()


def print_result(query):
    result = connection.execute(query)
    for row in result:
        print(row)


def count_missing(table_name):
    for columnn, in connection.execute("""select column_name from information_schema.columns where table_name = '""" + table_name + """' """):
        print(f'{columnn}:')
        print_result('''
            SELECT count(*) - count(''' + columnn + ''')
            FROM ''' + table_name + '''
        ''')


def get_counts(name, relation, title):
    query1 = '''
    SELECT count(distinct ''' + name + ''') FROM ''' + relation + ''';
    '''

    query2 = '''
    SELECT type, avg(CASE WHEN count IS NULL THEN 0 ELSE count END) avg
    FROM articles
    LEFT OUTER JOIN (
        SELECT article_index, count(''' + name + ''') count FROM ''' + relation + '''
        GROUP BY article_index
    ) distinct_counts
    ON article_index = index
    GROUP BY type
    ORDER BY avg;
    '''

    print(title + ':')
    print_result(query1)
    print_result(query2)
    print()


print('Statistics for FakeNewsCorpus:')

print('Number of articles in the FakeNewsCorpus dataset:')
print_result('''
SELECT count(*)
FROM articles
''')

print("Number of articles grouped by type:")
result = connection.execute('''
SELECT type, count(*) count
FROM articles
GROUP BY type
ORDER BY count
''')

fig, ax = plt.subplots()
ax.set_yscale('log')

labels = []
for i, row in enumerate(result):
    print(f'{row[0]}: {row[1]}')
    plt.bar(i, row[1], color='dodgerblue')
    labels.append(row[0])

fig.autofmt_xdate()
plt.xticks(range(len(labels)), labels)
plt.xlabel('Type')
plt.ylabel('Number of articles')
plt.show()


result = connection.execute(r'''
SELECT type, (LENGTH(cleaned_content) - LENGTH(replace(cleaned_content, ' ', ''))) + 1
FROM articles
''')


dict = {}

fig, ax = plt.subplots()

for row in result:
    lst = dict.get(row[0])
    if lst:
        lst.append(row[1])
    else:
        dict.update({row[0]: [row[1]]})

fig.autofmt_xdate()

labels = sorted(dict.keys(), key=lambda x: median(dict[x]))

plt.violinplot([dict[x] for x in labels], showextrema=False, quantiles=[[0.25, 0.5, 0.75] for x in labels])
plt.xticks(range(1, len(labels) + 1), labels)
plt.xlabel('Type')
plt.ylabel('Number of words')

plt.show()


print('Missing values by column:')
count_missing('articles')

print("Number of distinct domains for articles:")
print_result('''
SELECT count(DISTINCT domain)
FROM articles
''')


print("Number of domains with more than one type of article:")
print_result('''
SELECT count(*) FROM
(
    SELECT domain, count(DISTINCT type) as distinct_types
    FROM articles
    GROUP BY domain
) A
WHERE distinct_types != 1
''')

print("Average length of articles grouped by type:")
print_result('''
SELECT type, avg(char_length(cleaned_content)) avg
FROM articles
GROUP BY type
ORDER BY avg
''')

get_counts('author_name', 'authors', "AUTHORS")
get_counts('keyword_name', 'keywords', "KEYWORDS")
get_counts('meta_keyword_name', 'meta_keywords', "META_KEYWORDS")
get_counts('tag_name', 'tags', "TAGS")

print('Statistics for WikiNews:')

print('Number of articles in the WikiNews dataset:')
print_result('''
SELECT count(*)
FROM wikinews_articles
''')

print('Missing values by column:')
count_missing('wikinews_articles')

print('Number of categories')
print_result('''
SELECT count(DISTINCT category_name)
FROM wikinews_categories
''')

connection.close()
