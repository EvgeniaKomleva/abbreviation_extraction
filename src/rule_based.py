import spacy
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.matcher import Matcher

spacy.prefer_gpu()

BRACKETED_RULE = [{"ORTH": "("}, {"TEXT": {"NOT_IN": [")"]}, "OP": "+"}, {"ORTH": ")"}]

def match_abbreviation(short_form: str, long_form: str):
    sf_index, lf_index = len(short_form) - 1, len(long_form) - 1

    while sf_index >= 0:
        curr_char = short_form[sf_index].lower()
        if not curr_char.isalnum():
            sf_index -= 1
            continue

        while (lf_index >= 0 and long_form[lf_index].lower() != curr_char) or (
            sf_index == 0 and lf_index > 0 and long_form[lf_index - 1].isalnum()
        ):
            lf_index -= 1

        if lf_index < 0:
            return None

        lf_index -= 1
        sf_index -= 1

    return long_form[lf_index + 1:]

def find_abbreviation_candidates(sentence, parentheses_words):
    abbreviation_candidates = []

    for _, parentheses_start, parentheses_end in parentheses_words:
        parenthesis_word = sentence[parentheses_start + 1 : parentheses_end - 1]
        short_len = len(parenthesis_word.text)

        valid = parenthesis_word.text[0].isalnum() and any(c.isalpha() for c in parenthesis_word.text)
        valid = (len(parenthesis_word.text.split()) <= 2) and valid
        valid = 2 <= short_len <= 10 and valid

        if valid:
            long_start = max(0, parentheses_start - min(short_len + 5, short_len * 2))
            long_form = sentence[long_start:parentheses_start]

            abbreviation_candidates.append((str(parenthesis_word), str(long_form)))

    return abbreviation_candidates

class SimpleAbbreviationExtractor:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_md")

        infix_re = compile_infix_regex(tuple(x for x in self.nlp.Defaults.infixes if "-|–|—|--|---|——|~" not in x))

        def custom_tokenizer(nlp):
            return Tokenizer(
                nlp.vocab,
                prefix_search=nlp.tokenizer.prefix_search,
                suffix_search=nlp.tokenizer.suffix_search,
                infix_finditer=infix_re.finditer,
                token_match=nlp.tokenizer.token_match,
                rules=nlp.Defaults.tokenizer_exceptions,
            )

        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("bracketed", [BRACKETED_RULE])

    def find_abbreviations(self, text):
        processed_text = self.nlp(text)
        all_abbreviations = []

        for sent in processed_text.sents:
            parentheses_words = self.matcher(sent)
            parentheses_with_candidates = find_abbreviation_candidates(sent, parentheses_words)

            for possible_solution in parentheses_with_candidates:
                abbreviation_definition = match_abbreviation(
                    possible_solution[0], possible_solution[1]
                )
                if abbreviation_definition is not None:
                    all_abbreviations.append(
                        np.array([sent.text, possible_solution[0], abbreviation_definition])
                    )
                    print('!!!', sent.text,' possible_solution[0]!!', possible_solution[0], 'abbreviation_definition!!',abbreviation_definition)

        return np.array(all_abbreviations)
