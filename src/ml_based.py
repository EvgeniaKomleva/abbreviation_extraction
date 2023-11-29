
from typing import Iterable, List, Tuple
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class BertAbbreviationExtractor:
    def __init__(self, base_name) -> None:
        #self.nlp = spacy.load(base_name)
        self.pipeline = pipeline(task='ner', model="surrey-nlp/roberta-large-finetuned-abbr")
    def find_abbreviations(self, text: str):

        
        #print("Text: ",text)
        #print(self.pipeline(text))

        
        """
            When given a body of text, this function returns a list of
            source sentences, abbreviations, and their definitions.
        Args:
            text (str): string in which to find the abbreviations

        Returns:
            np.ndarray: An array of elements where each element
                gives an abbreviation along with its in-place sentence
                and its definition.
        """
        # np.array([sent.text, possible_solution[0], abbreviation_definition])
        # !!! The serum nickel concentration, fasting blood glucose (FPG) , fasting insulin (FIns) and glycosylated hemoglobin (HbA1c) were measured in the contact group and the control group.  
        # possible_solution[0]!! FIns 
        # abbreviation_definition!! fasting insulin
        #text = "The serum nickel concentration, fasting blood glucose (FPG) , fasting insulin (FIns) and glycosylated hemoglobin (HbA1c) were measured in the contact group and the control group.  "
        processed_text = self.pipeline(text)
        print('processed_text',processed_text)
        all_abbreviations = []
        long_form = ''
        abbreviation =''
        prev_tag = ''
        prev_start = 0
        prev_end = 0
        print('Text:', text)
        if len(processed_text)==0:
            return np.array([])
        current_entity = processed_text[0]
        concatenated_entities = []
        for next_entity in processed_text[1:]:
            if current_entity['entity'][2:] == next_entity['entity'][2:] and current_entity['end'] == next_entity['start']:
                current_entity['word'] += next_entity['word']
                current_entity['end'] = next_entity['end']
            else:
                concatenated_entities.append(current_entity)
                current_entity = next_entity


        concatenated_entities.append(current_entity)
        print('concatenated_entities', concatenated_entities)
        found_abbreviations = heuristic_abbreviation_match(concatenated_entities)
        if found_abbreviations:
                for found_pair in found_abbreviations:
                    all_abbreviations.append(np.array([text, *found_pair]))
        #for entity in processed_text:
            #print(entity)
            

        """if prev_tag in entity['entity'] and (prev_end == entity['start'] or prev_end +1 == entity['start'] ):
                if prev_tag =="LF":
                    long_form += entity['word'].replace('Ġ', ' ')
                else:
                    abbreviation += entity['word'].replace('Ġ', ' ')
            else:
                if abbreviation !='' and :
                    all_abbreviations.append(np.array([text,abbreviation, long_form ]))
                    print("ADD:!", abbreviation,"LF:", long_form, "ENT: ",entity )
                    long_form=''
                    abbreviation=''

            prev_tag = entity['entity'][2:]   """ 
                #prev_tag = 'AC'
        """if (long_form!='' and abbreviation!= '' and prev_tag not in entity['entity']) or (prev_end +2 <entity['start'] and long_form!='' and abbreviation!= ''):
                all_abbreviations.append(np.array([text,abbreviation, long_form ]))
                print("ADD:!", abbreviation, long_form, entity )
                abbreviation = ''
                long_form = ''
            if 'LF' in entity['entity']:
                long_form += entity['word'].replace('Ġ', ' ')
                #print(long_form)
                #prev_tag = 'LF'
            if 'AC' in entity['entity']:
                abbreviation += entity['word'].replace('Ġ', ' ')
                #print(abbreviation)
            prev_tag = entity['entity'][2:]
            prev_end = entity['end']
            #print('prev_tag',prev_tag)
        if long_form!='' and abbreviation!= '':
            all_abbreviations.append(np.array([text,abbreviation, long_form ]))
            print("ADD:!", abbreviation, long_form )"""
        return np.array(all_abbreviations)

def heuristic_abbreviation_match(found_entities):
    """
    Simple heuristc that matches short forms with a long form that appears
    previously in the string provided there is no other short form in between

    i.e - matches adjacent short/long forms

    Args:
        found_entities (List[spacy.tokens.span.Span}): Ordered list of short/long forms in
        order that they appear in text

    Returns:
        List: List of paired short/long forms
    """
    matches = []
    print(found_entities)

    for i in range(len(found_entities) - 1, 0, -1):
        #print(found_entities[i]['entity'])
        if "AC" in found_entities[i]['entity']  and "LF" in found_entities[i - 1]['entity']:
            matches.append([found_entities[i], found_entities[i - 1]])
    print('matches',matches)
    print('____________________________')
    return matches