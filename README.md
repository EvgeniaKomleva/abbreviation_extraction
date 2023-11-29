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


