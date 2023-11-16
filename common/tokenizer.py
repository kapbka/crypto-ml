from typing import List

import spacy

IGNORED = {'for', 'you', 'the', 'this', 'that', 'these', 'those', 'there', 'had', 'has', 'have',
           'and', 'with', 'not', 'its', 'your', 'where'}
EXCLUDED_TAGS = {"ADP", "AUX", "SYM", "NUM"}
nlp = spacy.load("en_core_web_sm")


def tokenize(text: str) -> List[str]:
    result = list()
    doc = nlp(text.strip().lower())

    for token in doc:
        if token.text.startswith('@') or token.is_stop:
            continue

        part = ''.join([i for i in token.lemma_.lower() if 97 <= ord(i) <= 122])
        if part.startswith('http') or not part:
            continue

        if token.pos_ not in EXCLUDED_TAGS and part not in IGNORED:
            result.append(part)

    return result
