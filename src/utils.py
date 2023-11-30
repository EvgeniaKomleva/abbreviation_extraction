from typing import Dict
import lxml.etree as le
import pandas as pd
from tqdm import tqdm
import numpy as np

XML_ELEMENTS = {
    "MedlineCitation/PMID": "PMID",
    "MedlineCitation/Article/ArticleTitle": "article_title",
    "MedlineCitation/Article/Abstract/AbstractText": "abstract",
}


def read_xml(xml_path: str) -> le._Element:
    xml_tree = le.parse(xml_path)
    return xml_tree.getroot()


def extract_element_text(element):
    return le.tostring(element, encoding="unicode", method="text", with_tail=False)


def load_articles(xml_path: str, target_elements: Dict[str, str] = XML_ELEMENTS) -> pd.DataFrame:
    xml_root = read_xml(xml_path)
    extracted_article_data = []

    for article_element in xml_root:
        article_data_as_text = [extract_element_text(article_element.find(element_path)) if article_element.find(element_path) is not None else None
                                for element_path in target_elements.keys()]
        extracted_article_data.append(article_data_as_text)

    df = pd.DataFrame(extracted_article_data, columns=target_elements.values())
    return df


def extract_abbreviations(extractor, cleaned_articles_df):
    abstracts_with_ids = cleaned_articles_df[["PMID", "abstract"]].to_numpy()
    abbreviations_with_ids = []

    for keyed_abstract in tqdm(abstracts_with_ids):
        abbrevs = extractor.find_abbreviations(keyed_abstract[1])

        if abbrevs.size == 0:
            continue
        abbrevs = np.insert(abbrevs, abbrevs.shape[1], keyed_abstract[0], axis=1)
        abbreviations_with_ids.append(abbrevs)

    abbreviations_with_ids = [
        item for sublist in abbreviations_with_ids for item in sublist
    ]

    combined_data = pd.DataFrame(
        abbreviations_with_ids, columns=["sentence", "short_form", "long_form", "PMID"]
    ).merge(cleaned_articles_df[["PMID", "article_title"]], how="left", on="PMID")[
        ["article_title", "PMID", "sentence", "short_form", "long_form"]
    ]

    return combined_data
