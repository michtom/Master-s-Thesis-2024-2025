from matplotlib import pyplot as plt
import seaborn as sns


def plot_predictions(y, y_pred) -> None:
  plt.figure(figsize=(12, 5))
  sns.scatterplot(x=y, y=y_pred)
  plt.xlabel('Actual price return')
  plt.ylabel('Predicted price return')
  plt.legend()
  plt.show()


def plot_history(hist) -> None:
  fig = plt.figure(figsize=(12, 5))

  ax = fig.add_subplot(122)
  sns.lineplot(x=range(len(hist.history['loss'])), y=hist.history['loss'], label='Training Loss', ax=ax)
  sns.lineplot(x=range(len(hist.history['val_loss'])), y=hist.history['val_loss'], label='Validation Loss', ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_predictions_over_time(y: pd.Series, y_pred: pd.Series) -> None:
    df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    }).dropna()

    plt.figure(figsize=(14, 5))

    sns.lineplot(
        data=df,
        x=df.index,
        y="Actual",
        label="Actual",
        linewidth=2
    )

    sns.lineplot(
        data=df,
        x=df.index,
        y="Predicted",
        label="Predicted",
        linewidth=2,
        linestyle="--"
    )

    plt.xlabel("Time")
    plt.ylabel("Price return")
    plt.title("Predictions vs Actual Values Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
