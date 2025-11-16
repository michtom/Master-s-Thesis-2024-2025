import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = "../../data/market_data"


def calculate_additional_features(df: pd.DataFrame,
                                  horizon: int = 1,
                                  window_functions_horizon: int = 24 * 12) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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



def _test() -> None:
    X, y = prepare_market_data_for_model('btc_merged.csv')
    print(X.head().to_string())
    print(y.head().to_string())


if __name__ == "__main__":
    _test()
