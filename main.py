
import argparse
from src.utils import clean_data, load_articles
from src.rule_based import SimpleAbbreviationExtractor
from src.utils import get_abbreviations
from src.ml_based import BertAbbreviationExtractor

def main(args):
    articles = load_articles(args.articles)
    print(articles)
    cleaned_articles = clean_data(articles)
    print(cleaned_articles)
    if args.rule_base:
        simple_abbreviation_extractor = SimpleAbbreviationExtractor()
        simple_abbreviation_output = get_abbreviations(
            simple_abbreviation_extractor, cleaned_articles
        )
        print(simple_abbreviation_output)
    if args.ml_base:
        bert = BertAbbreviationExtractor('EvgeniaKomleva/roberta-large-finetuned-abbr-finetuned-ner')#'surrey-nlp/en_abbreviation_detection_roberta_lar')
        bert_abbreviation_output = get_abbreviations(
            bert, cleaned_articles
        )
        print(bert_abbreviation_output)
        bert_abbreviation_output.to_csv(args.out_rule_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abbreviation Extraction")
    parser.add_argument(
        "--articles", default="./data/articles.xml", help="Path to the articles XML file"
    )
    parser.add_argument(
        "--rule_base", default=False, help="Get output from rule-based approach "
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
