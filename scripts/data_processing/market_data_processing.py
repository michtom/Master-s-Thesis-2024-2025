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
    df["news_count"] = news_count

    news_mean = news_mean.reindex(df.index).fillna(0.0).shift(1)

    minutes_per_bar = pd.to_timedelta("5min").seconds / 60

    for t in windows_hours:
        hl = t / 2
        hl_bars = hl * 60 / minutes_per_bar
        col_name = f"sentiment_ewm_{t}h"
        df[col_name] = news_mean.ewm(halflife=hl_bars, adjust=False).mean().shift(1)
    return df.dropna()



def _test() -> None:
    df = pd.read_csv("/Users/mtomczyk/Downloads/articles_labeled.csv")
    X, y = prepare_market_data_for_model('btc_merged.csv', horizon=4*12)
    print(X.head().to_string())
    print(y.head().to_string())
    windows_hours = (4, 6, 8, 10, 12, 2)
    result = add_sentiment_features_from_articles(X, df, windows_hours=windows_hours)
    #result = result[result['sentiment_ewm_36h'] > 0]
    print(result.head(25).to_string())
    cols = [f"sentiment_ewm_{i}h" for i in windows_hours]
    print(result[cols].corrwith(y))
    for h in []:  # steps ahead (5min * h)
        y_h = y # or however you define your label
        print(f"\nHorizon: {h} steps:")
        print(result[cols].iloc[:-h].corrwith(y_h.iloc[:-h]))
    #print(X.head().to_string())
    #print(y.head().to_string())


if __name__ == "__main__":
    _test()
