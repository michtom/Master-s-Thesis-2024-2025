import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt


def evaluate_predictions(predictions: np.ndarray, labels: np.ndarray,
                         target_names: list[str]=["negative", "neutral", "positive"]) -> dict[str, float]:
  accuracy = accuracy_score(labels, predictions)
  macro_f1 = f1_score(labels, predictions, average="macro")
  weighted_f1 = f1_score(labels, predictions, average="weighted")

  print("Accuracy:", accuracy)
  print("Macro F1:", macro_f1)
  print("Weighted F1:", weighted_f1)

  print("\nClassification report:")
  print(classification_report(labels, predictions, target_names=target_names))

  print("\nConfusion matrix:")
  cm = confusion_matrix(labels, predictions)
  print(cm)
  fig, ax = plt.subplots(figsize=(6, 5))
  im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

  plt.colorbar(im, ax=ax)
  ax.set(
    xticks=np.arange(len(target_names)),
    yticks=np.arange(len(target_names)),
    xticklabels=target_names,
    yticklabels=target_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix'
  )
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(
              j, i, format(cm[i, j], "d"),
              ha="center",
              va="center",
              color="white" if cm[i, j] > thresh else "black"
          )

  plt.tight_layout()
  plt.show()
  return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}