import matplotlib.pyplot as plt
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_metric, load_dataset
from sklearn.metrics import confusion_matrix
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

run = wandb.init(
    project="abbreviation_extraction",
)
class NERTrainer:
    def __init__(self, model_checkpoint, task="ner", batch_size=16):
        self.model_checkpoint = model_checkpoint
        self.task = task
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, add_prefix_space=True)
        self.label_list = None
        self.metric = load_metric("seqeval")
        self.model = None
        self.args = None
        self.data_collator = None
        self.trainer = None

    def load_dataset(self, dataset_name="surrey-nlp/PLOD-filtered"):
        datasets = load_dataset(dataset_name)
        self.label_list = datasets["train"].features[f"{self.task}_tags"].feature.names
        return datasets

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        label_all_tokens = True
        for i, label in enumerate(examples[f"{self.task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_datasets(self, datasets):
        tokenized_datasets = datasets.map(self.tokenize_and_align_labels, batched=True)
        return tokenized_datasets

    def initialize_model(self):
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.label_list))

    def initialize_training_args(self, output_dir, eval_steps=7000, save_steps=35000, num_train_epochs=6):
        self.args = TrainingArguments(
            output_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_total_limit=3,
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            weight_decay=0.001,
            save_steps=save_steps,
            report_to="wandb",
            push_to_hub=True,
            metric_for_best_model='f1',
            load_best_model_at_end=True
        )

    def initialize_data_collator(self):
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    def initialize_trainer(self, tokenized_train_dataset, tokenized_eval_dataset, early_stopping_patience=3):
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience)]
        )

    def train_model(self):
        self.trainer.train()

    def evaluate_model(self):
        self.trainer.evaluate()

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[self.label_list[p] for (p, l) in zip(
            prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(
            prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    def predict_and_evaluate(self, tokenized_test_dataset):
        predictions, labels, _ = self.trainer.predict(tokenized_test_dataset)
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[self.label_list[p] for (p, l) in zip(
            prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[self.label_list[l] for (p, l) in zip(
            prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = self.metric.compute(
            predictions=true_predictions, references=true_labels)
        return true_labels, true_predictions, results

    def save_and_push_model(self, model_name):
        self.trainer.save_model(model_name)
        self.trainer.push_to_hub()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, figsize=(10, 10)):
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        plt.savefig('output.png')
        sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='',
                    ax=ax).figure.savefig('file.png')


def train():
    # Set your model checkpoint and output directory
    model_checkpoint = "surrey-nlp/roberta-large-finetuned-abbr"
    output_dir = f"{model_checkpoint.split('/')[-1]}-finetuned-ner"

    ner_trainer = NERTrainer(model_checkpoint)
    datasets = ner_trainer.load_dataset()
    tokenized_datasets = ner_trainer.prepare_datasets(datasets)

    ner_trainer.initialize_model()
    ner_trainer.initialize_training_args(output_dir)
    ner_trainer.initialize_data_collator()
    ner_trainer.initialize_trainer(
        tokenized_datasets["train"], tokenized_datasets["validation"])

    ner_trainer.train_model()
    ner_trainer.evaluate_model()

    true_labels, true_predictions, results = ner_trainer.predict_and_evaluate(
        tokenized_datasets["test"])

    # Save and push the model to the Hugging Face Model Hub
    ner_trainer.save_and_push_model(output_dir)

    # Plot confusion matrix
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    true_predictions_flat = [item for sublist in true_predictions for item in sublist]
    NERTrainer.plot_confusion_matrix(true_labels_flat, true_predictions_flat)


if __name__ == "__main__":
    train()
