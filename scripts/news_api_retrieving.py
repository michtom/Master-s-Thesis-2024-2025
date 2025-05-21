import logging
import os
from typing import Dict, List
import pandas as pd
import datetime


from finlight_client import FinlightApi
from config import credentials


COLUMNS = ['source', 'title', 'summary', 'authors', 'publishDate', 'sentiment', 'confidence', 'article_id']


def fetch_latest_crypto_articles_data(folder_path: str, config: Dict[str, str],
                                      queries: List[str], page_size: int = 100, pages: int = 5) -> None:
    if len(queries) * pages > 40:
        raise Exception('Trying to send too many requests (mind the monthly rate limits!')
    logging.info('starting retrieving data from finlighten.me API:')
    client = FinlightApi(config)
    df_list = []
    # get timestamp in microseconds to obtain a unique id
    timestamp = datetime.datetime.now().timestamp() * 1000000
    current_date = datetime.datetime.now().strftime("%d_%m_%Y_%H")
    full_articles = {}
    counter = int(timestamp)
    for query in queries:
        for page in range(1, pages + 1):
            try:
                resp = client.articles.get_extended_articles(params={"query": query, "pageSize": page_size,
                                                                     'page': page, 'language': 'en'})
            except Exception as e:
                logging.error(f'could not get data from {query} page {page}: {e}')
                continue
            articles = resp.get('articles', [])
            for article in articles:
                counter += 1
                article['article_id'] = counter
                full_articles[counter] = article['content']
            articles_reduced = [{k: v for k, v in article.items() if k in COLUMNS} for article in articles]
            df = pd.DataFrame(articles_reduced)
            df['query'] = query
            df_list.append(df)
    final_df = pd.concat(df_list, ignore_index=True)
    logging.info(f'number of retrieved articles: {len(final_df)}')
    final_df = final_df.drop_duplicates(subset=['title'])
    logging.info(f'number of unique articles retrieved: {len(final_df)}')
    full_articles_df = pd.DataFrame(full_articles.items(), columns=['article_id', 'text'])
    full_articles_df = full_articles_df[full_articles_df['article_id'].isin(final_df['article_id'])]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    full_articles_df.to_parquet(f'{folder_path}/full_articles_{current_date}.parquet', index=False)
    final_df.to_csv(f'{folder_path}/articles_df_{current_date}.csv', index=False)
    logging.info('retrieving ended successfully')


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    folder_path = '../data/finlighten_news'
    config = {'api_key': credentials.api_key}
    queries = ['crypto', 'bitcoin', 'ethereum', 'solana', 'blockchain', 'tether', 'usd']
    page_size = 100
    pages = 5
    fetch_latest_crypto_articles_data(folder_path, config, queries, page_size, pages)


if __name__ == '__main__':
    main()
