import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = "../../data/market_data"


def calculate_additional_features(df: pd.DataFrame,
                                  horizon: int = 1,
                                  window_functions_horizon: int = 24 * 12) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df = df.sort_values('timestamp').set_index('timestamp')

    df['price_t-1'] = df['price'].shift(1)
    df['log_return_t-1'] = np.log(df['price'] / df['price_t-1'])

    df['log_return_t-h'] = np.log(df['price'].shift(-horizon) / df['price'])
    df['s_t'] = df['log_return_t-1'].rolling(window_functions_horizon, min_periods=window_functions_horizon).std()
    df['z_H'] = df['log_return_t-h'] / ((df['s_t'] + 1e-12) * np.sqrt(horizon))

    ema_30 = df['price'].ewm(span=2*12, adjust=False).mean()
    ema_4h = df['price'].ewm(span=4*12, adjust=False).mean()
    df['price_over_ema4h'] = df['price'] / ema_4h - 1.0
    df['price_over_ema2h'] = df['price'] / ema_30 - 1.0

    df['volatility_4h'] = df['log_return_t-1'].rolling(4*12, min_periods=4*12).std()
    df['volatility_wh'] = df['log_return_t-1'].rolling(window_functions_horizon,
                                                       min_periods=window_functions_horizon).std()

    df['volume_mean'] = df['volume'].rolling(window_functions_horizon,
                                          min_periods=window_functions_horizon).mean()
    df['volume_std'] = df['volume'].rolling(window_functions_horizon,
                                          min_periods=window_functions_horizon).std()
    df['volume_z'] = (df['volume'] - df['volume_mean']) / (df['volume_std'] + 1e-12)

    feature_cols = [
        'price_over_ema4h', 'price_over_ema2h',
        'volatility_4h', 'volatility_wh',
        'volume_z', 's_t'
    ]
    df[feature_cols] = df[feature_cols].shift(1)

    keep_cols = feature_cols + ['z_H', 'log_return_t-h', 'log_return_t-1']
    df = df[keep_cols].dropna()
    return df


def prepare_market_data_for_model(
    filename: str,
    data_dir: str = DEFAULT_DATA_DIR,
    selected_features: list[str] | None = None,
    horizon: int = 1,
    window_functions_horizon: int = 24 * 12
) -> tuple[pd.DataFrame, pd.Series]:

    df = pd.read_csv(f"{data_dir}/{filename}")
    df = calculate_additional_features(df, horizon, window_functions_horizon)

    target_feature = 'log_return_t-h'

    default_features = [col for col in df.columns if col not in [target_feature, 'log_return_t-h', 'z_H']]
    features = default_features if selected_features is None else selected_features
    X = df[features].copy()
    y = df[target_feature].copy()
    return X, y


label2id = {"negative": -1, "neutral": 0, "positive": 1}
# so that no sentiment is treated as neutral sentiment (different than classes in BERT)

def add_sentiment_features_from_articles(
        df: pd.DataFrame,
        news: pd.DataFrame,
        windows_hours=(2, 6, 24),
) -> pd.DataFrame:
    news = news.copy()
    df = df.copy()
    news['publishDate'] = pd.to_datetime(news['publishDate'], format="mixed")
    news = news.sort_values('publishDate').set_index('publishDate')
    news["labels"] = news["label"].apply(lambda x: label2id.get(x, 0))
    news_mean = news["labels"].resample("5min").mean()
    news_count = news["label"].resample("5min").count()
    news_count = news_count.reindex(df.index).fillna(0.0)
    df['news_any'] = (news_count > 0).astype(float).shift(1)

    news_mean = news_mean.reindex(df.index).fillna(0.0)

    minutes_per_bar = 5

    for t in windows_hours:
        bars = int(t * 60 / minutes_per_bar)
        df[f'sentiment_mean_{t}h'] = news_mean.rolling(bars, min_periods=1).mean().shift(1)
        count_rolling = news_count.rolling(bars, min_periods=1).sum()
        df[f'news_count_{t}h'] = count_rolling.shift(1)
        df[f'has_news_{t}h'] = (count_rolling > 0).astype(float).shift(1)
    return df.dropna()


def add_sentiments_from_reddit(df: pd.DataFrame, comments:pd.DataFrame,
                               windows_hours=(2, 6, 24)) -> pd.DataFrame:
    comments = comments.copy()
    df = df.copy()
    comments["created_utc"] = pd.to_datetime(comments["created_utc"])
    comments["created_utc"] = comments["created_utc"].dt.tz_localize("UTC")
    comments = comments.sort_values("created_utc").set_index("created_utc")
    comments["labels"] = comments["label"].apply(lambda x: label2id.get(x, 0))
    sentiment = comments["labels"].resample("5min").mean().reindex(df.index).fillna(0.0)
    count = comments["labels"].resample("5min").count().reindex(df.index).fillna(0.0)

    df["has_comment"] = (count > 0).astype(float).shift(1)
    df["comment_count"] = np.log1p(count).shift(1)
    minutes = 5
    for t in windows_hours:
        bars = int((5 * 60) / minutes)
        sentiment_rolling = sentiment.rolling(bars, min_periods=1).mean()
        counts_rolling = count.rolling(bars, min_periods=1).mean()

        df[f"reddit_sentiment_{t}h"] = sentiment_rolling.shift(1)
        df[f"reddit_counts_{t}h"] = counts_rolling.shift(1)
    return df.dropna()


def _test() -> None:
    df = pd.read_csv("/Users/mtomczyk/Downloads/articles_labeled.csv")
    X, y = prepare_market_data_for_model('btc_merged.csv', horizon=4*12)
    windows_hours = (2, 6, 12, 42)
    result = add_sentiment_features_from_articles(X, df, windows_hours=windows_hours)
    print(result.head(30).to_string())
    comments = pd.read_csv("/Users/mtomczyk/Downloads/comments_labeled_non_finetuned.csv")
    print(comments.shape)
    result = add_sentiments_from_reddit(result, comments=comments)
    print(result.head(25).to_string())

if __name__ == "__main__":
    _test()
