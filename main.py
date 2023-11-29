
import argparse

from src.ml_based import BertAbbreviationExtractor
from src.rule_based import RuleBaseModel
from src.utils import extract_abbreviations, load_articles
from src.train import train

def main(args):
    articles = load_articles(args.articles)
    cleaned_articles = articles.dropna(subset=["abstract"])
    if args.rule_base:
        rule_base_model = RuleBaseModel()
        rule_base_output = extract_abbreviations(rule_base_model, cleaned_articles)
        rule_base_output.to_csv(args.out_rule_path, sep="\t", index=False)
    if args.ml_base:
        bert = BertAbbreviationExtractor(
            'EvgeniaKomleva/roberta-large-finetuned-abbr-finetuned-ner')
        bert_abbreviation_output = extract_abbreviations(
            bert, cleaned_articles
        )
        bert_abbreviation_output.to_csv(args.out_bert_path, sep="\t", index=False)
    if args.train:
        train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abbreviation Extraction")
    parser.add_argument(
        "--articles", default="./data/articles.xml", help="Path to the articles XML file"
    )
    parser.add_argument(
        "--rule_base", default=True, help="Get output from rule-based approach "
    )
    parser.add_argument(
        "--ml_base", default=True, help="Get output from ml-based approach "
    )
    parser.add_argument(
        "--train", default=False, help="Train ml model "
    )
    parser.add_argument(
        "--out_bert_path", default='./data/output_bert.tsv', help="Train ml model "
    )
    parser.add_argument(
        "--out_rule_path", default='./data/output_rule.tsv', help="Train ml model "
    )
    args = parser.parse_args()
    main(args)
