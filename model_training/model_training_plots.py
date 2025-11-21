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
