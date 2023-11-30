from typing import List, Tuple

import numpy as np
from transformers import pipeline


class BertAbbreviationExtractor:
    def __init__(self, model_name, task='ner') -> None:
        self.pipeline = pipeline(task=task, model=model_name)

    def find_abbreviations(self, text: str) -> np.ndarray:
        processed_text = self.pipeline(text)
        if len(processed_text) == 0:
            return np.array([])
        concatenated_entities = self.concatenate_entities(processed_text)
        found_abbreviations = self.find_matches(concatenated_entities)
        return np.array([np.array([text, *found_pair]) for found_pair in found_abbreviations])

    @staticmethod
    def concatenate_entities(entities: List[dict]) -> List[dict]:
        current_entity = entities[0]
        concatenated_entities = []

        for next_entity in entities[1:]:
            if current_entity['entity'][2:] == next_entity['entity'][2:] and current_entity['end'] + 2 >= next_entity['start']:
                current_entity['word'] += next_entity['word']
                current_entity['end'] = next_entity['end']
            else:
                concatenated_entities.append(current_entity)
                current_entity = next_entity

        concatenated_entities.append(current_entity)
        return concatenated_entities

    @staticmethod
    def find_matches(found_entities: List[dict]) -> List[List[str]]:
        matches = []

        for i in range(len(found_entities) - 1, 0, -1):
            if "AC" in found_entities[i]['entity'] and "LF" in found_entities[i - 1]['entity']:
                matches.append([found_entities[i]['word'].replace('Ġ', ' '),
                               found_entities[i - 1]['word'].replace('Ġ', ' ')])

        return matches
