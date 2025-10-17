import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = "../../data/market_data"


def prepare_market_data_for_model(
        filename: str,
        data_dir: str = DEFAULT_DATA_DIR,
        selected_features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(f'{data_dir}/{filename}')
    df = calculate_additional_features(df)
    df = df.dropna()
    target_feature = 'log_return'
    features = [col for col in df.columns if
                col not in [target_feature] + ['timestamp', 'date', 'price_shift', 'symbol']]
    if selected_features is not None:
        features = selected_features
    X = df[features]
    y = df[target_feature]
    return X, y


def calculate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.index = df['timestamp']
    df = df.drop(columns=['timestamp'])
    df = df.sort_index()
    df['price_shift'] = df['price'].shift(-1)
    df['log_return'] = np.log(df['price_shift'] / df['price'])
    df['volatility_1h'] = np.log(df['price'] / df['price'].shift(1)).rolling(pd.Timedelta('1h')).std()
    df['volatility_4h'] = np.log(df['price'] / df['price'].shift(1)).rolling(pd.Timedelta('4h')).std()
    df['volatility_12h'] = np.log(df['price'] / df['price'].shift(1)).rolling(pd.Timedelta('12h')).std()
    df['vwap_1h'] = (df['volume'] * df['price']).rolling(pd.Timedelta('1h')).sum() / df['volume'].rolling(pd.Timedelta('1h')).sum()
    df['vwap_4h'] = (df['volume'] * df['price']).rolling(pd.Timedelta('4h')).sum() / df['volume'].rolling(pd.Timedelta('4h')).sum()
    df['vwap_12h'] = (df['volume'] * df['price']).rolling(pd.Timedelta('12h')).sum() / df['volume'].rolling(pd.Timedelta('12h')).sum()
    df['ema_1'] = df['price'].ewm(span=1).mean()
    df['ema_4h'] = df['price'].ewm(halflife='4h', times=df.index).mean()
    df['ema_30min'] = df['price'].ewm(halflife='30min', times=df.index).mean()
    return df


def _test() -> None:
    X, y = prepare_market_data_for_model('btc_merged.csv')
    print(X.head())
    print(y.head())


if __name__ == "__main__":
    _test()
