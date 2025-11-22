import logging
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    pipeline
)
from datasets import Dataset


class MLMUnlabeledDataTrainer:
    def __init__(self, final_checkpoint, mlm_checkpoint, models_path, device="cuda",
                 epochs_num: int = 5):
        self.final_checkpoint = f"{models_path}/{final_checkpoint}"
        self.mlm_checkpoint = f"{models_path}/{mlm_checkpoint}"
        self.device = device
        self.base_model_name = "yiyanghkust/finbert-tone"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs_num = epochs_num
        self.tokenizer = None
        self.mlm_model = None
        self.final_model = None
        self.data_collator = None
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def init_data_collator(self, initial_model: bool = False) -> None:
        checkpoint = self.mlm_checkpoint if not initial_model else self.base_model_name
        self.init_mlm_model(checkpoint)
        if self.tokenizer is None:
            self.init_tokenizer()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=0.15,
            mlm=True
        )

    def init_mlm_model(self, checkpoint: str) -> None:
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    def init_final_model(self, checkpoint: str) -> None:
        self.final_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=3,
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def tokenize(self, example):
        if self.tokenizer is None:
            self.init_tokenizer()
        return self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    def train_mlm_model(self, dataset: Dataset, initial_model: bool = False) -> None:
        self.init_data_collator(initial_model)
        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir="./finbert-mlm-crypto",
            overwrite_output_dir=True,
            num_train_epochs=self.epochs_num,
            per_device_train_batch_size=16,
            save_steps=500,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=50,
            report_to="none",
            eval_strategy="no",
        )
        trainer = Trainer(
            model=self.mlm_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=self.data_collator,
        )
        trainer.train()
        trainer.save_model(self.mlm_checkpoint)

    @property
    def pseudo_pipeline(self):
        if self.tokenizer is None:
            self.init_tokenizer()
        return pipeline(
            "sentiment-analysis",
            model=self.base_model_name,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=128
        )

    def assign_pseudo_labels(self, dataset: Dataset, dataset_path : str) -> Dataset:
        texts = dataset["train"]["text"]
        pseudo_labels = []
        pseudo_texts = []
        pipeline = self.pseudo_pipeline
        for t in texts:
            out = pipeline(t)[0]
            if out["score"] >= 0.9:
                pseudo_texts.append(t)
                pseudo_labels.append(out["label"])
        label_ids = [self.label2id[l.lower()] for l in pseudo_labels]
        pseudo_ds = Dataset.from_dict({"text": pseudo_texts, "labels": label_ids})
        pseudo_ds.to_csv(dataset_path)
        return pseudo_ds

    def train_final_model(self, dataset: Dataset, pseudo_dataset_path: str) -> None:
        try:
            self.init_final_model(self.mlm_checkpoint)
        except Exception as e:
            self.logger.error(e)
            return
        pseudo_ds = self.assign_pseudo_labels(dataset, dataset_path=pseudo_dataset_path)
        tokenized_dataset = pseudo_ds.map(self.tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir="./finbert-sentiment-pseudo",
            num_train_epochs=self.epochs_num,
            per_device_train_batch_size=16,
            eval_strategy="no",
            report_to="none",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=False,
        )
        trainer = Trainer(
            model = self.final_model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        trainer.train()
        self.final_model.save_pretrained(self.final_checkpoint)

    def train_final_model_labeled_data(self, dataset: Dataset) -> None:
        try:
            self.init_final_model(self.mlm_checkpoint)
        except Exception as e:
            self.logger.error(e)
            return
        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir="./finbert-sentiment-pseudo",
            num_train_epochs=self.epochs_num,
            per_device_train_batch_size=16,
            eval_strategy="no",
            report_to="none",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=False,
        )
        trainer = Trainer(
            model = self.final_model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        trainer.train()
        self.final_model.save_pretrained(self.final_checkpoint)

    def predict_sentiment(self, text: str, device: str, max_length: int=128):
        self.final_model.to(device)
        self.final_model.eval()

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.final_model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            pred_label = self.id2label[pred_id]

        return {
            "label_id": pred_id,
            "label": pred_label,
            "probs": probs[0].cpu().tolist()
        }