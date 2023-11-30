# Abbreviation Extraction

This is a tool for extracting abbreviations from articles using both rule-based and machine learning-based approaches.

## Installation

Make sure you have Python installed. Then, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

To extract abbreviations from articles, you can use the `main.py` script. You have the following options:

```bash
python main.py --articles ./data/articles.xml --rule_base --ml_base --train --out_bert_path ./data/output_bert.tsv --out_rule_path ./data/output_rule.tsv
```
## Options

- `--articles`: Path to the articles XML file (default: "./data/articles.xml").
- `--rule_base`: Flag to enable the rule-based approach (default: True).
- `--ml_base`: Flag to enable the machine learning-based approach (default: True).
- `--train`: Flag to train the machine learning model (default: False).
- `--out_bert_path`: Path to save the output of the machine learning-based approach (default: './data/output_bert.tsv').
- `--out_rule_path`: Path to save the output of the rule-based approach (default: './data/output_rule.tsv').


## Methods

### Rule-based solution

The rule-based approach for abbreviation detection works by employing a combination of pattern matching and heuristics to identify potential abbreviations within a given text. The find_abbreviations method of the RuleBaseModel class processes the input text using spaCy.
For each sentence in the processed text, it utilizes the Matcher to identify words within parentheses (parentheses_words).
It then uses find_abbreviation_candidates to identify potential abbreviation candidates based on the criteria mentioned.
For each potential solution, it employs the match_abbreviation function to determine if it is a valid abbreviation. If valid, the abbreviation and its details are appended to the result. 

The RuleBaseModel class uses a spaCy Matcher to identify words enclosed within parentheses in the text. The specific rule for this matching is defined as:
```python
self.rule = [{"ORTH": "("}, {"TEXT": {"NOT_IN": [")"]}, "OP": "+"}, {"ORTH": ")"}]
```
This rule looks for sequences of words enclosed within parentheses.

### ML-based solution


1. Dataset
The dataset used in this solution is loaded using the datasets library, specifically the load_dataset function. The dataset is named "surrey-nlp/PLOD-filtered" and it is used for training, validation, and testing the model.

The dataset likely contains text documents, and each document may have annotated information about named entities, including abbreviations. The dataset is then split into training, validation, and test sets.

2. Task: Named Entity Recognition (NER)
The task at hand is Named Entity Recognition (NER), a natural language processing task where the goal is to identify and classify entities (e.g., names of people, organizations, locations) in text. In this specific case, the NER task is specialized for detecting abbreviations in text.

3. Model: RoBERTa for Token Classification
The model used for this NER task is RoBERTa (Robustly optimized BERT approach), a pre-trained transformer-based model for natural language understanding. The model is fine-tuned for token classification, where each token in the input text is assigned a label indicating whether it belongs to an abbreviation entity.

### Results
Results ML model for "surrey-nlp/PLOD-filtered" dataset: 
#### Evaluation Metrics
eval/f1=0.97835
eval/loss=0.09133
eval/precision=0.98004
eval/recall=0.97667

#### Training Statistics
train/epoch=6.0
train/global_step=42246
train/learning_rate=0.0
train/loss=0.0197
train/total_flos=2.059382411725857e+17
train/train_loss=0.0458
train/train_runtime=17401.0239s
train/train_samples_per_second=38.843
train/train_steps_per_second=2.428






