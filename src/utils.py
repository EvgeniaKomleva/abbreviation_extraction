from typing import Dict
import lxml.etree as et
import pandas as pd
from tqdm import tqdm
import numpy as np

TARGET_ELEMENTS = {
        "MedlineCitation/PMID": "PMID",
        "MedlineCitation/Article/ArticleTitle": "article_title",
        "MedlineCitation/Article/Abstract/AbstractText": "abstract",
    }

def read_articles(read_location: str) -> et._Element:
    """Reads xml file from disk and returns the root node

    Args:
        read_location (str): Relative path to the file

    Returns:
        et._Element: XML root element
    """
    xml_tree = et.parse(read_location)
    return xml_tree.getroot()
def load_articles(
    read_location: str, target_elements: Dict[str, str] = TARGET_ELEMENTS
) -> pd.DataFrame:
    """Loads all articles from an XML file and returns a dataframe
    with their IDs, title, and abstract information

    Args:
        read_location (str): The file location to read articles from

        target_elements (dict(str, str)): A dictionary of XML element paths to extract
            for each article. The keys are the XML element paths, the values are the
            names for the dataframe's columns corresponding to each XML element.
    Returns:
        pandas.DataFrame: The dataframe containing all articles from the file
    """

    xml_root = read_articles(read_location)
    extracted_article_data = []
    for article in xml_root:
        article_data_as_text = []

        # Find the desired nodes and for each one convert their data to simple text
        for element in map(article.find, target_elements.keys()):
            if element is not None:  # xml tostring functionality is not nullsafe
                element = et.tostring(
                    element, encoding="unicode", method="text", with_tail=False
                )
            article_data_as_text.append(element)

        extracted_article_data.append(article_data_as_text)

    df = pd.DataFrame(extracted_article_data, columns=target_elements.values())

    return df


def clean_data(articles: pd.DataFrame) -> pd.DataFrame:
    """Removes non-NLP related errors from the articles data
        In this narrow use-case, the only errors are rows with missing abstracts,
        which are removed

    Args:
        articles (pd.DataFrame): DataFrame containing article information

    Returns:
        pd.DataFrame: DataFrame containing processed article information
    """
    return articles.dropna(subset=["abstract"])


def get_abbreviations(extractor, cleaned_articles):
    """Wrapper function that abstracts the logic for:
        - making our input articles' data match with the abbreviation-extractors
        - Iterating across each abstract
        - returning a properly formatted dataframe ready for output.

    Args:
        cleaned_articles (pd.DataFrame): The input articles in a dataframe form

    Returns:
        pd.DataFrame: A DataFrame containing sentence-per-row entries where each row
        matches the task's given schema. Covering every article from the input.
    """

    abstracts_with_ids = cleaned_articles[["PMID", "abstract"]].to_numpy()  # for speed
    abbreviations_with_ids = []

    for keyed_abstract in tqdm(abstracts_with_ids):
        # Get the abbreviations for a single abstract, tracked per sentence
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
    ).merge(cleaned_articles[["PMID", "article_title"]], how="left", on="PMID")[
        ["article_title", "PMID", "sentence", "short_form", "long_form"]
    ]

    return combined_data