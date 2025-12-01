import random
from enum import Enum, auto
import json

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from model_training.loss_functions import *
from model_training.price_model_validation import *
from model_training.model_training_plots import *


class ModelType(Enum):
    LSTM = auto()
    GRU = auto()
    ARIMA = auto()
    DUMMY = auto()

class LossFunction(Enum):
    MSE = auto()
    HUBER = auto()
    HUBER_WITH_VERMATCH = auto()
    EXPECTILE_VAR = auto()


def create_lstm(num_features, seq_length=6, num_neurons=64, dropout_rate=0.3,
                num_layers: int = 2) -> tf.keras.Model:
    layers = [tf.keras.layers.Input(shape=(seq_length, num_features))]
    for i in range(num_layers):
        if i == num_layers - 1:
            layers.append(tf.keras.layers.LSTM(num_neurons, activation='tanh', dropout=dropout_rate,
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-4), ))
        else:
            layers.append(tf.keras.layers.LSTM(num_neurons, activation='tanh', dropout=dropout_rate,
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                               return_sequences=True))
        layers.append(tf.keras.layers.LayerNormalization())
    layers.extend([tf.keras.layers.Dense(64, activation="elu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation=None)])
    model = tf.keras.Sequential(layers)
    return model


def create_gru(num_features, seq_length=6, num_neurons=512, dropout_rate=0.3,
               num_layers: int = 2) -> tf.keras.Model:
    layers = [tf.keras.layers.Input(shape=(seq_length, num_features))]
    for i in range(num_layers):
        if i == num_layers - 1:
            layers.append(tf.keras.layers.GRU(num_neurons, activation='tanh', dropout=dropout_rate,
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4), ))
        else:
            layers.append(tf.keras.layers.GRU(num_neurons, activation='tanh', dropout=dropout_rate,
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                              return_sequences=True))
        layers.append(tf.keras.layers.LayerNormalization())
    layers.extend([tf.keras.layers.Dense(64, activation="elu"),
                   tf.keras.layers.Dropout(0.1),
                   tf.keras.layers.Dense(1, activation=None)])
    model = tf.keras.Sequential(layers)
    return model


def create_arima(endog, exog=None, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)) -> SARIMAX:
    return SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=False,
    )


def create_dummy(seq_length=6, num_neurons=512, dropout_rate=0.3) -> tf.keras.Model:
    pass


def build_sequences(X_df, y_array, seq_len) -> tuple[np.ndarray, np.ndarray]:
    X_np = X_df.values.astype(np.float32)
    y_np = y_array.values.astype(np.float32).reshape(-1, 1)

    n, f = X_np.shape
    idx_end = np.arange(seq_len - 1, n)

    X_seq = np.stack([X_np[i - seq_len + 1:i + 1, :] for i in idx_end], axis=0)
    y_cut = y_np[idx_end]

    return X_seq, y_cut

def make_json_serializable(obj) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    else:
        return obj

def save_dict_to_json(data_dict, filename) -> None:
  data = make_json_serializable(data_dict)
  with open(f"{filename}.json", "w") as f:
    json.dump(data, f, indent=4)


def train_price_prediction_model(
        X: pd.DataFrame, y: pd.Series,
        model_type: ModelType, split: TimeSeriesSplit,
        filename: str,
        dropout: float = 0.15, learning_rate: float = 1e-3,
        seq_length: int = 48, num_neurons: int = 256, batch_size: int = 128,
        loss_funtion: LossFunction = LossFunction.EXPECTILE_VAR,
        epochs: int = 100, num_layers: int = 2,
) -> tuple[dict, list[tf.keras.callbacks.History]]:
    np.random.seed(120)
    random.seed(120)
    tf.random.set_seed(120)
    scores = {"mse": [], "mae": [], "da": [], "sr": [],
              "r_squared": [], 'corr': [],
              "y_pred": [], "y_test": []}
    histories = []

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]

    i = 0
    for train_index, test_index in split.split(X):
        i += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_raw, y_test_raw = y.iloc[train_index].values.reshape(-1, 1), y.iloc[test_index].values.reshape(-1, 1)

        x_scaler = StandardScaler().fit(X_train.values)
        X_train[:] = x_scaler.transform(X_train.values)
        X_test[:] = x_scaler.transform(X_test.values)

        X_train_seq, y_train_seq_raw = build_sequences(X_train, pd.Series(y_train_raw.ravel(), index=X_train.index),
                                                       seq_length)
        X_test_seq, y_test_seq_raw = build_sequences(X_test, pd.Series(y_test_raw.ravel(), index=X_test.index),
                                                     seq_length)

        y_scaler = StandardScaler().fit(y_train_seq_raw)
        y_train = y_scaler.transform(y_train_seq_raw).astype(np.float32)
        y_test = y_scaler.transform(y_test_seq_raw).astype(np.float32)

        if model_type == ModelType.LSTM:
            model = create_lstm(num_features=X_train_seq.shape[2],
                                seq_length=seq_length,
                                num_neurons=num_neurons, dropout_rate=dropout,
                                num_layers=num_layers)
        elif model_type == ModelType.GRU:
            model = create_gru(num_features=X_train_seq.shape[2],
                               seq_length=seq_length,
                               num_neurons=num_neurons, dropout_rate=dropout,
                               num_layers=num_layers)
        elif model_type == ModelType.ARIMA:
            model = create_arima(endog=y_train_raw.ravel(), exog=X_train, order=(2, 1, 2))
            results = model.fit(disp=False)

            pred = results.get_forecast(steps=len(y_test_raw), exog=X_test)
            y_pred_raw = pred.predicted_mean.values
            y_test_raw_flat = y_test_raw.ravel()

            scores["y_pred"].append(y_pred_raw)
            scores["y_test"].append(y_test_raw_flat)
            scores["mse"].append(mean_squared_error(y_test_raw_flat, y_pred_raw))
            scores["mae"].append(mean_absolute_error(y_test_raw_flat, y_pred_raw))
            scores["da"].append(directional_accuracy(y_test_raw_flat, y_pred_raw))
            scores["sr"].append(sharpe_ratio(y_test_raw_flat, y_pred_raw))
            scores["r_squared"].append(r2_score(y_test_raw_flat, y_pred_raw))
            scores['corr'].append(calculate_correlation(y_test_raw_flat, y_pred_raw))

            print(f"fold {i} | mse: {scores['mse'][-1]:.6g} | mae: {scores['mae'][-1]:.6g} "
                  f"da: {scores['da'][-1]:.3f} | sr: {scores['sr'][-1]:.3f}")
            continue
        else:
            raise ValueError("Use LSTM/GRU here; ARIMA/DUMMY not implemented.")
        match loss_funtion:
            case LossFunction.MSE:
                loss = "mse"
            case LossFunction.HUBER:
                loss = tf.keras.losses.Huber(delta=1.0)
            case LossFunction.HUBER_WITH_VERMATCH:
                loss = huber_with_varmatch(delta=1.0, lam=0.01)
            case LossFunction.EXPECTILE_VAR:
                loss = loss_expectile_var(tau=0.5, lam=0.01)
            case _:
                loss = "mse"

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=loss
        )

        history = model.fit(
            X_train_seq, y_train,
            validation_data=(X_test_seq, y_test),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            shuffle=False,
        )
        histories.append(history)

        y_pred_std = model.predict(X_test_seq).reshape(-1, 1)
        y_pred_raw = y_scaler.inverse_transform(y_pred_std).ravel()
        y_test_raw_flat = y_test_seq_raw.ravel()

        scores["y_pred"].append(y_pred_raw)
        scores["y_test"].append(y_test_raw_flat)
        scores["mse"].append(mean_squared_error(y_test_raw_flat, y_pred_raw))
        scores["mae"].append(mean_absolute_error(y_test_raw_flat, y_pred_raw))
        scores["da"].append(directional_accuracy(y_test_raw_flat, y_pred_raw))
        scores["sr"].append(sharpe_ratio(y_test_raw_flat, y_pred_raw))
        scores["r_squared"].append(r2_score(y_test_raw_flat, y_pred_raw))
        scores["corr"].append(calculate_correlation(y_test_raw_flat, y_pred_raw))

        print(f"fold {i} | mse: {scores['mse'][-1]:.6g} | mae: {scores['mae'][-1]:.6g} "
              f"da: {scores['da'][-1]:.3f} | sr: {scores['sr'][-1]:.3f}")

    largest_r_squared = np.argmax(scores["r_squared"])
    y_pred = scores["y_pred"][largest_r_squared]
    y_test = scores["y_test"][largest_r_squared]
    plot_predictions(y_test.flatten(), y_pred)
    if model_type != ModelType.ARIMA:
        plot_history(histories[largest_r_squared])
    save_dict_to_json(scores, filename)
    print(f"succesfully saved training's output to {filename}")
    print(f"mean MSE: {np.mean(scores['mse'])}")
    print(f"mean MAE: {np.mean(scores['mae'])}")
    print(f"mean DA: {np.mean(scores['da'])}")
    print(f"mean SR: {np.mean(scores['sr'])}")
    print(f"mean R^2: {np.mean(scores['r_squared'])}")
    print(f"mean Pearson correlation: {np.mean(scores['corr'])}")
    return scores, histories
