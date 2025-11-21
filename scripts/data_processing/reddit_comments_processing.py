import pandas as pd
import re
import html
import numpy as np
from fontTools.subset import subset

from scripts.data_processing.financial_news_data_processing import load_all_data


emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE
)

def clean_comment(text, remove_emojis=False):
    if not isinstance(text, str):
        return ""

    # decode HTML entities (&amp; -> &, etc.)
    text = html.unescape(text)

    # remove reddit quoted text ">"
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # remove markdown links "[text](url)"
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # remove weird control characters
    text = re.sub(r"[\r\n\t]", " ", text)

    # optionally remove emojis
    if remove_emojis:
        text = emoji_pattern.sub("", text)

    # remove non-standard unicode symbols (optional)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_process_reddit_comments(folder: str, min_comment_length: int = 20,
                                     percentage_best_comments: float = 0.3) -> pd.DataFrame:
    df = load_all_data("reddit_comments", folder, "parquet")
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['created_date'] = df['created_utc'].dt.date
    df['comment'] = df['comment'].apply(lambda x: clean_comment(x, remove_emojis=False))
    df['comment_length'] = df['comment'].str.len()
    df = df[df['comment_length'] > min_comment_length]
    df = df[~df['comment'].str.startswith("http")]
    df['upvotes_log'] = np.log1p(df["upvotes"])
    df['upvote_pct_day'] = df.groupby('created_date')['upvotes_log'].rank(pct=True)
    df = df[df['upvote_pct_day'] > 1-percentage_best_comments]
    df = df.drop_duplicates(subset=['comment', 'post_title'])
    df = df.drop(columns=['subreddit', 'post_title', 'created_date', 'comment_length'])
    return df


if __name__ == "__main__":
    print(load_and_process_reddit_comments("../../data/reddit_comments").head(10).to_string())
