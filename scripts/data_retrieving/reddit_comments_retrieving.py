import praw
import logging
from typing import List
import pandas as pd
import datetime

from config import credentials


SUBREDDITS = ['CryptoCurrency']


def fetch_latest_reddit_comments(path: str, client: praw.Reddit, subreddits: List[str], min_comment_upvotes: int = 5,
                                 time_threshold_hours: int = 24) -> None:
    logging.info('fetching comments from reddit')
    current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
    current_date = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d-%H')
    comments = []
    for subreddit in subreddits:
        try:
            subreddit_client = client.subreddit(subreddit)
            for submission in subreddit_client.hot(limit=50):
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    comment_time = datetime.datetime.fromtimestamp(comment.created_utc, datetime.timezone.utc).timestamp()
                    if (comment_time - current_time) / 60 * 60 <= time_threshold_hours and comment.score >= min_comment_upvotes:
                        comments.append({
                            'subreddit': subreddit,
                            'post_title': submission.title,
                            'comment': comment.body,
                            'upvotes': comment.score,
                            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc)
                        })
        except Exception as e:
            logging.error(f'could not retrieve data from Reddit: {e}')
    df = pd.DataFrame(comments)
    df.to_parquet(f'{path}/reddit_comments_{current_date}.parquet', index=False)


def main() -> None:
    path = '../../data/reddit_comments/'
    logging.basicConfig(level=logging.INFO)
    client = praw.Reddit(client_id=credentials.reddit_client_id, client_secret=credentials.reddit_secret,
                         user_agent=credentials.reddit_agent)
    fetch_latest_reddit_comments(path, client, SUBREDDITS)


if __name__ == '__main__':
    main()
