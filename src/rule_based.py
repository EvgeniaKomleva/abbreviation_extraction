import numpy as np
import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex


def match_abbreviation(short_form: str, long_form: str):
    short_index, long_index = len(short_form) - 1, len(long_form) - 1

    while short_index >= 0:
        curr_char = short_form[short_index].lower()
        if not curr_char.isalnum():
            short_index -= 1
            continue

        while (long_index >= 0 and long_form[long_index].lower() != curr_char) or (
            short_index == 0 and long_index > 0 and long_form[long_index - 1].isalnum()
        ):
            long_index -= 1

        if long_index < 0:
            return None

        long_index -= 1
        short_index -= 1

    return long_form[long_index + 1:]


def find_abbreviation_candidates(sentence, parentheses_words):
    abbreviation_candidates = []

    for _, start, end in parentheses_words:
        word_inside_parentheses = sentence[start + 1: end - 1]
        short_length = len(word_inside_parentheses.text)

        valid = word_inside_parentheses.text[0].isalnum() and any(
            c.isalpha() for c in word_inside_parentheses.text)
        valid = (len(word_inside_parentheses.text.split()) <= 2) and valid
        valid = 2 <= short_length <= 10 and valid

        if valid:
            long_start = max(0, start - min(short_length + 5, short_length * 2))
            long_form = sentence[long_start:start]

            abbreviation_candidates.append(
                (str(word_inside_parentheses), str(long_form)))

    return abbreviation_candidates


class RuleBaseModel:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_md")

        infix_re = compile_infix_regex(
            tuple(x for x in self.nlp.Defaults.infixes if "-|–|—|--|---|——|~" not in x))

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
        self.rule = [{"ORTH": "("}, {"TEXT": {"NOT_IN": [")"]},
                                     "OP": "+"}, {"ORTH": ")"}]
        self.matcher.add("bracketed", [self.rule])

    def find_abbreviations(self, text):
        processed_text = self.nlp(text)
        all_abbreviations = []

        for sent in processed_text.sents:
            parentheses_words = self.matcher(sent)
            parentheses_with_candidates = find_abbreviation_candidates(
                sent, parentheses_words)

            for possible_solution in parentheses_with_candidates:
                abbreviation_definition = match_abbreviation(
                    possible_solution[0], possible_solution[1]
                )
                if abbreviation_definition is not None:
                    all_abbreviations.append(
                        np.array([sent.text, possible_solution[0],
                                 abbreviation_definition])
                    )

        return np.array(all_abbreviations)
