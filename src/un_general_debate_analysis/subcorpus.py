"""Pull the sentences that mention a lexicon back out of the speeches.

``speeches_features.csv`` says *how many* sentences of a speech mention
technology; this module gives you the sentences themselves. That is what
makes the lexicon-based metrics checkable ("show me 20 sentences the
promise/peril dictionary fired on") and what a keyness comparison needs
("which words does a well-connected country use when it talks about
technology, and a poorly-connected one?").

It re-uses the tokeniser, lemmatiser and phrase matcher from
``preprocessing`` so a sentence is labelled here exactly as it was counted
there.

    from un_general_debate_analysis.subcorpus import lexicon_sentences

    texts = pd.read_csv(OUT_DIR / "speeches_text.csv.gz")
    tech = lexicon_sentences(texts.query("year >= 1990"), "technology")

Roughly 5 seconds per 1,000 speeches, so about a minute for the whole
corpus. It runs on one core by default because a process pool started from
a notebook kernel is fragile; pass ``workers=8`` from a script if you need
the speed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

from un_general_debate_analysis.preprocessing import (
    classify_sentiment,
    lexicon_matcher,
    normalise_tokens,
    sentiment_analyser,
)


def _label_speech(text: str) -> list[tuple[str, tuple[str, ...], float]]:
    """Split one speech into (sentence, lexicons it mentions, VADER score)."""
    mwe, term_to_lexicons = lexicon_matcher()
    sia = sentiment_analyser()
    labelled = []
    for sentence in sent_tokenize(str(text)):
        tokens = mwe.tokenize(normalise_tokens(word_tokenize(sentence)))
        lexicons = {
            lexicon
            for token in tokens
            if token in term_to_lexicons
            for lexicon in term_to_lexicons[token]
        }
        if lexicons:
            labelled.append(
                (sentence, tuple(sorted(lexicons)), sia.polarity_scores(sentence)["compound"])
            )
    return labelled


def lexicon_sentences(
    texts: pd.DataFrame,
    lexicon: str = "technology",
    text_column: str = "text_clean",
    workers: int = 1,
) -> pd.DataFrame:
    """Every sentence of ``texts`` that mentions ``lexicon``.

    ``texts`` is ``speeches_text.csv.gz`` (or a slice of it) and must keep
    its ``country_code`` and ``year`` columns. The result has one row per
    sentence: the keys, the sentence, its VADER compound score, the tuple of
    lexicons it matched, and a 0/1 column per lexicon for easy filtering.
    """
    payload = texts[text_column].astype(str).tolist()

    if workers > 1 and len(payload) > 200:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            labelled = list(executor.map(_label_speech, payload, chunksize=32))
    else:
        labelled = [_label_speech(text) for text in payload]

    keys = texts[["country_code", "year"]].reset_index(drop=True)
    records = [
        (keys.at[i, "country_code"], keys.at[i, "year"], sentence, score, lexicons)
        for i, sentences in enumerate(labelled)
        for sentence, lexicons, score in sentences
        if lexicon in lexicons
    ]
    frame = pd.DataFrame(
        records, columns=["country_code", "year", "sentence", "compound", "lexicons"]
    )
    if frame.empty:
        return frame

    frame["sentiment"] = frame["compound"].map(classify_sentiment)
    present = sorted({name for names in frame["lexicons"] for name in names})
    for name in present:
        frame[name] = frame["lexicons"].map(lambda names, n=name: int(n in names))
    return frame.reset_index(drop=True)
