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


TEST_FRACTION = 0.2


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
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx].sort_index()
    y = y.loc[common_idx].sort_index()
    n = len(X)
    n_test = int(n * TEST_FRACTION)
    X_trainval = X.iloc[:-n_test].copy()
    y_trainval = y.iloc[:-n_test].copy()
    X_test_final = X.iloc[-n_test:].copy()
    y_test_final = y.iloc[-n_test:].copy()
    scores = {
        "cv_mse": [],
        "cv_mae": [],
        "cv_da": [],
        "cv_sr": [],
        "cv_r_squared": [],
        "cv_corr": [],
        "cv_y_pred": [],
        "cv_y_true": [],
    }
    scores_test: dict[str, float] = {}
    histories: list[tf.keras.callbacks.History] = []

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]

    i = 0
    for train_index, test_index in split.split(X_trainval):
        i += 1
        X_train, X_val = X_trainval.iloc[train_index], X_trainval.iloc[test_index]
        y_train_raw, y_val_raw = y_trainval.iloc[train_index].values.reshape(-1, 1), y_trainval.iloc[test_index].values.reshape(-1, 1)

        x_scaler = StandardScaler().fit(X_train.values)
        X_train_scaled = x_scaler.transform(X_train.values)
        X_val_scaled = x_scaler.transform(X_val.values)

        X_train_scaled_df = pd.DataFrame(
            X_train_scaled,
            index=X_train.index,
            columns=X_train.columns
        )
        X_val_scaled_df = pd.DataFrame(
            X_val_scaled,
            index=X_val.index,
            columns=X_val.columns
        )
        if model_type == ModelType.ARIMA:
            model = create_arima(
                endog=y_train_raw.ravel(),
                exog=X_train_scaled_df,
                order=(2, 1, 2)
            )
            results = model.fit(disp=False)

            pred = results.get_forecast(
                steps=len(y_val_raw),
                exog=X_val_scaled_df
            )
            y_pred_raw = pred.predicted_mean.values
            y_val_raw_flat = y_val_raw.ravel()

            scores["cv_y_pred"].append(y_pred_raw)
            scores["cv_y_true"].append(y_val_raw_flat)
            scores["cv_mse"].append(mean_squared_error(y_val_raw_flat, y_pred_raw))
            scores["cv_mae"].append(mean_absolute_error(y_val_raw_flat, y_pred_raw))
            scores["cv_da"].append(directional_accuracy(y_val_raw_flat, y_pred_raw))
            scores["cv_sr"].append(sharpe_ratio(y_val_raw_flat, y_pred_raw))
            scores["cv_r_squared"].append(r2_score(y_val_raw_flat, y_pred_raw))
            scores["cv_corr"].append(calculate_correlation(y_val_raw_flat, y_pred_raw))

            print(
                f"[CV fold {i}] mse: {scores['cv_mse'][-1]:.6g} | "
                f"mae: {scores['cv_mae'][-1]:.6g} | "
                f"da: {scores['cv_da'][-1]:.3f} | "
                f"sr: {scores['cv_sr'][-1]:.3f}"
            )
            continue

        y_train_series = pd.Series(
            y_train_raw.ravel(),
            index=X_train_scaled_df.index
        )
        y_val_series = pd.Series(
            y_val_raw.ravel(),
            index=X_val_scaled_df.index
        )

        X_train_seq, y_train_seq_raw = build_sequences(
            X_train_scaled_df,
            y_train_series,
            seq_length
        )
        X_val_seq, y_val_seq_raw = build_sequences(
            X_val_scaled_df,
            y_val_series,
            seq_length
        )

        y_scaler = StandardScaler().fit(y_train_seq_raw)
        y_train = y_scaler.transform(y_train_seq_raw).astype(np.float32)
        y_val = y_scaler.transform(y_val_seq_raw).astype(np.float32)

        if model_type == ModelType.LSTM:
            model = create_lstm(
                num_features=X_train_seq.shape[2],
                seq_length=seq_length,
                num_neurons=num_neurons,
                dropout_rate=dropout,
                num_layers=num_layers
            )
        elif model_type == ModelType.GRU:
            model = create_gru(
                num_features=X_train_seq.shape[2],
                seq_length=seq_length,
                num_neurons=num_neurons,
                dropout_rate=dropout,
                num_layers=num_layers
            )
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
            validation_data=(X_val_seq, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            shuffle=False,
        )
        histories.append(history)

        y_pred_std = model.predict(X_val_seq, verbose=0).reshape(-1, 1)
        y_pred_raw = y_scaler.inverse_transform(y_pred_std).ravel()
        y_val_raw_flat = y_val_seq_raw.ravel()

        scores["cv_y_pred"].append(y_pred_raw)
        scores["cv_y_true"].append(y_val_raw_flat)
        scores["cv_mse"].append(mean_squared_error(y_val_raw_flat, y_pred_raw))
        scores["cv_mae"].append(mean_absolute_error(y_val_raw_flat, y_pred_raw))
        scores["cv_da"].append(directional_accuracy(y_val_raw_flat, y_pred_raw))
        scores["cv_sr"].append(sharpe_ratio(y_val_raw_flat, y_pred_raw))
        scores["cv_r_squared"].append(r2_score(y_val_raw_flat, y_pred_raw))
        scores["cv_corr"].append(calculate_correlation(y_val_raw_flat, y_pred_raw))

        print(
            f"[CV fold {i}] mse: {scores['cv_mse'][-1]:.6g} | "
            f"mae: {scores['cv_mae'][-1]:.6g} | "
            f"da: {scores['cv_da'][-1]:.3f} | "
            f"sr: {scores['cv_sr'][-1]:.3f}"
        )

    final_history = None
    if X_test_final is not None and y_test_final is not None:
        # Train once on FULL trainval block and evaluate on final test
        X_tr = X_trainval.copy()
        y_tr = y_trainval.copy().values.reshape(-1, 1)
        X_te = X_test_final.copy()
        y_te_raw = y_test_final.copy().values.reshape(-1, 1)

        # feature scaler from all training+val
        x_scaler_final = StandardScaler().fit(X_tr.values)
        X_tr_scaled = x_scaler_final.transform(X_tr.values)
        X_te_scaled = x_scaler_final.transform(X_te.values)

        X_tr_scaled_df = pd.DataFrame(
            X_tr_scaled,
            index=X_tr.index,
            columns=X_tr.columns
        )
        X_te_scaled_df = pd.DataFrame(
            X_te_scaled,
            index=X_te.index,
            columns=X_te.columns
        )

        if model_type == ModelType.ARIMA:
            model_final = create_arima(
                endog=y_tr.ravel(),
                exog=X_tr_scaled_df,
                order=(2, 1, 2)
            )
            results_final = model_final.fit(disp=False)

            pred_final = results_final.get_forecast(
                steps=len(y_te_raw),
                exog=X_te_scaled_df
            )
            y_pred_test_raw = pred_final.predicted_mean.values
            y_test_raw_flat = y_te_raw.ravel()
        else:
            # sequences for final model
            y_tr_series = pd.Series(y_tr.ravel(), index=X_tr_scaled_df.index)
            y_te_series = pd.Series(y_te_raw.ravel(), index=X_te_scaled_df.index)

            X_tr_seq, y_tr_seq_raw = build_sequences(
                X_tr_scaled_df,
                y_tr_series,
                seq_length
            )
            X_te_seq, y_te_seq_raw = build_sequences(
                X_te_scaled_df,
                y_te_series,
                seq_length
            )

            y_scaler_final = StandardScaler().fit(y_tr_seq_raw)
            y_tr_scaled = y_scaler_final.transform(y_tr_seq_raw).astype(np.float32)
            y_te_scaled = y_scaler_final.transform(y_te_seq_raw).astype(np.float32)

            # create model again
            if model_type == ModelType.LSTM:
                model_final = create_lstm(
                    num_features=X_tr_seq.shape[2],
                    seq_length=seq_length,
                    num_neurons=num_neurons,
                    dropout_rate=dropout,
                    num_layers=num_layers
                )
            elif model_type == ModelType.GRU:
                model_final = create_gru(
                    num_features=X_tr_seq.shape[2],
                    seq_length=seq_length,
                    num_neurons=num_neurons,
                    dropout_rate=dropout,
                    num_layers=num_layers
                )
            else:
                raise ValueError("Use LSTM/GRU here; ARIMA/DUMMY not implemented.")

            # same loss as before
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

            model_final.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=1.0
                ),
                loss=loss
            )

            final_history = model_final.fit(
                X_tr_seq,
                y_tr_scaled,
                validation_data=(X_te_seq, y_te_scaled),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=False,
                verbose=0,
            )

            y_pred_test_std = model_final.predict(X_te_seq, verbose=0).reshape(-1, 1)
            y_pred_test_raw = y_scaler_final.inverse_transform(y_pred_test_std).ravel()
            y_test_raw_flat = y_te_seq_raw.ravel()

        scores_test["test_mse"] = mean_squared_error(y_test_raw_flat, y_pred_test_raw)
        scores_test["test_mae"] = mean_absolute_error(y_test_raw_flat, y_pred_test_raw)
        scores_test["test_da"] = directional_accuracy(y_test_raw_flat, y_pred_test_raw)
        scores_test["test_sr"] = sharpe_ratio(y_test_raw_flat, y_pred_test_raw)
        scores_test["test_r_squared"] = r2_score(y_test_raw_flat, y_pred_test_raw)
        scores_test["test_corr"] = calculate_correlation(y_test_raw_flat, y_pred_test_raw)

        print(
            f"[FINAL TEST] mse: {scores_test['test_mse']:.6g} | "
            f"mae: {scores_test['test_mae']:.6g} | "
            f"da: {scores_test['test_da']:.3f} | "
            f"sr: {scores_test['test_sr']:.3f} | " 
            f"r_squared: {scores_test['test_r_squared']:.3f} | "
            f"corr: {scores_test['test_corr']:.3f}"
        )
        plot_predictions(y_test_raw_flat, y_pred_test_raw)
        if model_type != ModelType.ARIMA and final_history is not None:
            plot_history(final_history)

    print("=== Cross-validation (validation folds) ===")
    print(f"mean CV MSE: {np.mean(scores['cv_mse'])}")
    print(f"mean CV MAE: {np.mean(scores['cv_mae'])}")
    print(f"mean CV DA: {np.mean(scores['cv_da'])}")
    print(f"mean CV SR: {np.mean(scores['cv_sr'])}")
    print(f"mean CV R^2: {np.mean(scores['cv_r_squared'])}")
    print(f"mean CV Pearson correlation: {np.mean(scores['cv_corr'])}")

    # Add test scores to scores dict if present
    scores.update(scores_test)
    if scores_test and X_test_final is not None:
        plot_predictions(y_test_raw_flat, y_pred_test_raw)
        if model_type != ModelType.ARIMA and final_history is not None:
            plot_history(final_history)
    save_dict_to_json(scores, filename)
    print(f"successfully saved training's output to {filename}")
    return scores, histories
