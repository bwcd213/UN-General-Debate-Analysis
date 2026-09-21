"""Preprocessing pipeline for the UN General Debate Corpus (UNGDC), 1946-2025.

It reads the raw speech files and metadata in ``data/raw/`` and writes
analysis-ready datasets to ``data/processed/``:

1. ``speeches_features.csv``
   One row per speech (= one country-year). Holds metadata (country, region,
   speaker and speaker role), text statistics, NLTK VADER sentiment, and,
   for every lexicon in ``lexicons.py``, both how often its terms occur
   (``<lexicon>_count``, counted in words) and how its sentences are
   coloured (``<lexicon>_positive_sentences`` and friends, counted in
    sentences). It also includes binary education and technology topic
    features, including average sentiment and same-sentence overlap.
   ``term_counts`` is a JSON dict of how often each lexicon term occurs in
   the speech (load it with ``json.loads``).
   Key: (country_code, year). This is the table to merge with external
   country-year datasets (World Happiness Report, trade data, Our World in
   Data, ...); see ``merge_country_year``.

    It is wide (one block of four columns per lexicon, plus the education and
    technology focus features). For the sentence-level view on its own, select
    ``META_COLUMNS + sentence_count_columns()``.

2. ``speeches_text.csv.gz``
   The cleaned, readable text and the lemmatised, stop-word-free tokens of
   each speech. Use it for word clouds, TF-IDF, topic models or as input to
   a predictive model. It is kept separate because it is large.

3. ``lexicon_terms_by_year.csv``
   How often each lexicon term was used each year. Shows which words drive a
   trend (e.g. did "climate" rise because of "paris agreement"?).

Run it from the project root with:

    uv run python -m un_general_debate_analysis.preprocessing
    uv run python -m un_general_debate_analysis.preprocessing --limit 300   # quick test

The NLP steps use NLTK: sentence and word tokenisation (Punkt/Treebank),
English stop words, the WordNet lemmatiser, a multi-word-expression tokeniser
for phrases, and the VADER sentiment analyser.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer, sent_tokenize, word_tokenize

from un_general_debate_analysis.lexicons import (
    CO_OCCURRENCE_PAIRS,
    LEXICONS,
    NEUTRAL_PHRASES,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# preprocessing.py lives in <project>/src/un_general_debate_analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# NLTK data the pipeline needs: {path used by nltk.data.find: package name}
NLTK_RESOURCES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet.zip": "wordnet",  # NLTK reads WordNet straight from the zip
    "corpora/omw-1.4.zip": "omw-1.4",
    "sentiment/vader_lexicon.zip": "vader_lexicon",
}

# Speech files are named like "USA_75_2020.txt": ISO3 code, session, year.
# Stray files in the archive (".DS_Store", ".Rhistory", ...) do not match.
FILENAME_PATTERN = re.compile(
    r"^(?P<country_code>[A-Z]+)_(?P<session>\d+)_(?P<year>\d{4})\.txt$"
)

# Session 80 (2025) was transcribed from audio, so it contains the presiding
# officer's introductions and thank-yous. We strip those sentences.
AUDIO_TRANSCRIBED_SESSIONS = {80}

# Speech codes for states that no longer exist are missing from the UNSD
# M49 table, so we add them by hand.
HISTORICAL_STATES = pd.DataFrame(
    [
        ("CSK", "Czechoslovakia", "Europe", "Eastern Europe"),
        ("DDR", "German Democratic Republic", "Europe", "Western Europe"),
        ("YUG", "Yugoslavia", "Europe", "Southern Europe"),
        ("YMD", "Democratic Yemen", "Asia", "Western Asia"),
        ("EU", "European Union", "Europe", "European Union"),
    ],
    columns=["country_code", "country", "region_name", "sub_region_name"],
)

# Procedural and ceremonial words that appear in almost every speech but say
# nothing about its topic. They are removed from ``tokens_clean`` only;
# lexicon counts still see the full text, so "united nations reform" matches.
UN_PROCEDURAL_WORDS = [
    "united", "nations", "general", "assembly", "president", "secretary",
    "session", "delegation", "excellency", "distinguished", "madam", "sir",
    "mr", "mrs", "ms", "ladies", "gentlemen", "thank", "also", "would",
    "must", "shall", "may", "us",
]

# Sentences that belong to the presiding officer rather than the speaker
# (only used for the audio-transcribed sessions). Examples:
#   "The Assembly will now hear an address by His Excellency ..."
#   "I request protocol to escort His Excellency ..."
#   "On behalf of the Assembly, I wish to thank the Prime Minister of ..."
PRESIDING_OFFICER_SENTENCE = re.compile(
    r"^(?:(?:and|now|so|as|excellency)[,\s]+)*"
    r"(?:the (?:general )?assembly (?:will|shall) (?:now )?hear"
    r"|the (?:general )?assembly (?:has|have) (?:just )?heard"
    r"|i (?:now )?give the floor"
    r"|i request (?:the )?protocol"
    r"|on behalf of the (?:general )?assembly,? i (?:wish to )?thank"
    r"|i (?:wish to )?thank the (?:vice|deputy|first|prime|president|minister"
    r"|federal|chair|head|king|crown|amir|emir|prince|permanent|foreign)"
    r"[^.]*? of (?!the (?:general )?assembly))",
    re.IGNORECASE,
)

# Every sentence is put in one of three classes with VADER's usual
# thresholds: positive at compound >= 0.05, negative at compound <= -0.05,
# neutral in between. Only the two outer classes get their own columns; the
# neutral ones are visible as the gap between them and the totals.
REPORTED_SENTIMENTS = ("positive", "negative")

# Every lexicon also gets a 0/1 presence flag and the mean VADER compound
# score of the sentences that mention it, e.g. ``technology`` and
# ``technology_sentiment``. Lexicon *pairs* listed in CO_OCCURRENCE_PAIRS get
# the same treatment for sentences that mention both, e.g.
# ``technology_frame_peril_sentences``.
SENTIMENT_LEXICONS: tuple[str, ...] = tuple(LEXICONS)

# The country-year dimensions every output table starts with.
META_COLUMNS = [
    "country_code", "country", "year", "session", "region_name", "sub_region_name",
    "speaker_name", "speaker_post", "speaker_role",
]

# Typographic characters normalised to plain ASCII before tokenising.
TYPOGRAPHIC_CHARS = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": " - ", "­": "", " ": " ",
    }
)


# ---------------------------------------------------------------------------
# Step 0: make sure the NLTK data is available
# ---------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Download any NLTK model or corpus that is not installed yet."""
    for resource_path, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource '{package}' ...")
            nltk.download(package, quiet=True)


# ---------------------------------------------------------------------------
# Step 1: load the raw data
# ---------------------------------------------------------------------------


def load_speeches(txt_dir: Path) -> pd.DataFrame:
    """Read every speech file into a DataFrame with key columns from the name."""
    rows = []
    for path in sorted(txt_dir.rglob("*.txt")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:  # skip stray files such as ".DS_Store-to-UTF-8.txt"
            continue
        rows.append(
            {
                "country_code": match["country_code"],
                "session": int(match["session"]),
                "year": int(match["year"]),
                # "utf-8-sig" also removes the byte-order mark some files start with
                "speech": path.read_text(encoding="utf-8-sig"),
            }
        )
    return pd.DataFrame(rows)


def load_country_metadata(path: Path) -> pd.DataFrame:
    """Load UNSD M49 country names, regions and development-group flags."""
    meta = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    meta = meta.rename(
        columns={
            "ISO-alpha3 Code": "country_code",
            "Country or Area": "country",
            "Region Name": "region_name",
            "Sub-region Name": "sub_region_name",
        }
    )[["country_code", "country", "region_name", "sub_region_name"]]

    return pd.concat([meta, HISTORICAL_STATES], ignore_index=True)


def categorise_post(post: object) -> str:
    """Map the free-text post of a speaker to a small set of roles.

    The raw column has ~270 spellings (e.g. "Minister for Foregn Affairs",
    "PRIME MINISTER", "Prime Minister and Minister for Finance, ..."). The
    first title in the string decides the category.
    """
    if not isinstance(post, str) or not post.strip():
        return "unknown"
    p = " ".join(post.lower().replace("foregn", "foreign").split())

    if re.match(r"(first )?(vice|deputy)[- ]?(president|prime minister|premier|chancellor)|crown prince", p):
        return "deputy_head_of_state_or_government"
    if re.match(r"prime minister|premier|head of government|chancellor|taoiseach"
                r"|president of the (council of ministers|government)", p):
        return "head_of_government"
    if re.match(r"(constitutional |interim |acting |state )?(president|head of state)|king|queen|(sovereign )?prince"
                r"|amir|emir|sultan|grand duke|captain regent|chairman of the presiden", p):
        return "head_of_state"
    if re.match(r"deputy minister|vice[- ]minister|minister of state", p):
        return "other_minister"
    if re.search(r"foreign|external affairs|secretary of state|relations with states", p):
        return "foreign_minister"
    if "minister" in p:
        return "other_minister"
    if re.search(r"representative|ambassador|delegation", p):
        return "un_diplomat"
    return "other"


def load_speakers(path: Path) -> pd.DataFrame:
    """Load speaker names and posts, with one row per country-year."""
    speakers = pd.read_excel(
        path, usecols=["Year", "ISO Code", "Name of Person Speaking", "Post"]
    ).rename(
        columns={
            "Year": "year",
            "ISO Code": "country_code",
            "Name of Person Speaking": "speaker_name",
            "Post": "speaker_post",
        }
    )
    speakers = speakers.dropna(subset=["country_code", "year"])
    speakers["year"] = speakers["year"].astype(int)
    speakers["country_code"] = speakers["country_code"].str.strip()
    speakers["speaker_name"] = speakers["speaker_name"].str.strip()
    speakers["speaker_post"] = speakers["speaker_post"].str.strip()
    speakers["speaker_role"] = speakers["speaker_post"].map(categorise_post)

    # A few country-years list two people. Keep the row that records a post,
    # so a merge on (country_code, year) does not duplicate speeches.
    speakers = (
        speakers.assign(has_post=speakers["speaker_post"].notna())
        .sort_values("has_post", ascending=False, kind="stable")
        .drop_duplicates(["country_code", "year"])
        .drop(columns="has_post")
    )
    return speakers


# ---------------------------------------------------------------------------
# Step 2: clean the raw text
# ---------------------------------------------------------------------------


def _join_hyphenated_line_break(match: re.Match[str]) -> str:
    """Repair words split across lines by the PDF layout ("in-\\nvestment").

    Join the halves only when WordNet knows the result. Otherwise keep the
    hyphen, because it was a real compound ("Syrian-\\nowned").
    """
    left, right = match.group(1), match.group(2)
    if wordnet.synsets((left + right).lower()):
        return left + right
    return f"{left}-{right}"


def clean_text(text: str) -> str:
    """Remove layout artefacts and return the speech as one line of text."""
    text = text.translate(TYPOGRAPHIC_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF page headers/footers on a line of their own: the UN job number with
    # a page number ("09-52228 18"), or the document symbol with the date
    # ("01/10/2015 A/70/PV.22")
    text = re.sub(r"(?m)^[ \t]*(?:\d{1,3}[ \t]+)?\d{2}-\d{5}(?:[ \t]+\d{1,3})?[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*(?:\d{2}/\d{2}/\d{4}[ \t]+)?A/\d+/PV\.\d+(?:[ \t]+\d{2}/\d{2}/\d{4})?[ \t]*$", "", text)
    # Older verbatim records number every paragraph ("155.\t May I begin ...")
    text = re.sub(r"(?m)^[ \t]*\d{1,4}\.[ \t]+", "", text)
    # Hard-wrapped PDFs split words across lines with a hyphen
    text = re.sub(r"(\w+)-\n(\w+)", _join_hyphenated_line_break, text)
    # Collapse line breaks and repeated spaces
    return re.sub(r"\s+", " ", text).strip()


def strip_presiding_officer(sentences: list[str], max_strip: int = 3) -> list[str]:
    """Drop the chair's intro and outro sentences from audio transcripts."""
    start = 0
    while start < min(max_strip, len(sentences)) and PRESIDING_OFFICER_SENTENCE.match(sentences[start]):
        start += 1
    end = len(sentences)
    while end > start and len(sentences) - end < max_strip and PRESIDING_OFFICER_SENTENCE.match(sentences[end - 1]):
        end -= 1
    return sentences[start:end]


# ---------------------------------------------------------------------------
# Step 3: NLP helpers (tokenise, lemmatise, match lexicons, sentiment)
# ---------------------------------------------------------------------------

_LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """NLTK's English stop-word list."""
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=None)
def lemmatise(word: str) -> str:
    """Reduce a word to its dictionary form with the WordNet lemmatiser.

    Every word is lemmatised as a noun ("nations" -> "nation"). Words ending
    in "-ed" are also lemmatised as verbs ("transformed" -> "transform").
    "-ing" words are left alone, otherwise "housing" would become "house" and
    "learning" would become "learn". The cache matters: the corpus has ~35M
    tokens but only ~100k distinct words.
    """
    lemma = _LEMMATIZER.lemmatize(word, pos="n")
    if lemma.endswith("ed"):
        lemma = _LEMMATIZER.lemmatize(lemma, pos="v")
    return lemma


def normalise_tokens(raw_tokens: Iterable[str]) -> list[str]:
    """Turn NLTK word tokens into lowercase lemmas.

    - Split tokens on anything that is not a letter, so "HIV/AIDS" becomes
      "hiv", "aids" and "COVID-19" becomes "covid".
    - Drop single letters and numbers.
    - Leave stop words as they are. The noun lemmatiser would otherwise turn
      "us" into "u" and "was" into "wa". Stop words are also needed to match
      phrases like "rule of law".
    """
    stop_words = english_stopwords()
    tokens = []
    for raw in raw_tokens:
        for part in re.findall(r"[^\W\d_]+", raw.lower()):
            if len(part) < 2:
                continue
            tokens.append(part if part in stop_words else lemmatise(part))
    return tokens


@lru_cache(maxsize=1)
def clean_stopwords() -> frozenset[str]:
    """Words removed from ``tokens_clean``: NLTK stop words plus UN boilerplate."""
    return english_stopwords() | frozenset(normalise_tokens(UN_PROCEDURAL_WORDS))


@lru_cache(maxsize=1)
def lexicon_matcher() -> tuple[MWETokenizer, dict[str, tuple[str, ...]]]:
    """Build the phrase tokeniser and a lookup from term to lexicon(s).

    Every lexicon term is normalised the same way as the speeches. Multi-word
    terms are registered with NLTK's MWETokenizer, which merges them into one
    token such as "climate_change".
    """
    phrases: list[tuple[str, ...]] = []
    term_to_lexicons: dict[str, set[str]] = defaultdict(set)

    for lexicon, terms in LEXICONS.items():
        for term in terms:
            tokens = normalise_tokens(word_tokenize(term))
            if len(tokens) > 1:
                phrases.append(tuple(tokens))
            term_to_lexicons["_".join(tokens)].add(lexicon)

    # Neutral phrases are merged too, but they are not in term_to_lexicons,
    # so they never count
    phrases.extend(tuple(normalise_tokens(word_tokenize(p))) for p in NEUTRAL_PHRASES)

    tokenizer = MWETokenizer(phrases, separator="_")
    return tokenizer, {term: tuple(sorted(lex)) for term, lex in term_to_lexicons.items()}


@lru_cache(maxsize=1)
def sentiment_analyser() -> SentimentIntensityAnalyzer:
    """NLTK's VADER sentiment model, loaded once per process."""
    return SentimentIntensityAnalyzer()


def classify_sentiment(compound: float) -> str:
    """Label one sentence with VADER's usual thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyse_speech(args: tuple[str, int]) -> dict:
    """Run the full NLP pipeline on one speech, sentence by sentence.

    This runs in a worker process, so it takes and returns plain Python
    objects. It returns the cleaned text, clean tokens, text statistics,
    sentiment, lexicon counts, the per-term lexicon counts, and how
    many sentences fall in each (lexicon, sentiment) cell.
    """
    raw_text, session = args

    # --- Cleaning and sentence segmentation (Punkt) ---
    sentences = sent_tokenize(clean_text(raw_text))
    if session in AUDIO_TRANSCRIBED_SESSIONS:
        sentences = strip_presiding_officer(sentences)

    mwe_tokenizer, term_to_lexicons = lexicon_matcher()
    # VADER is tuned for short texts, so score each sentence and summarise.
    # The compound score runs from -1 (very negative) to +1 (very positive).
    sia = sentiment_analyser()

    words: list[str] = []  # every lemma of the speech
    merged_words: list[str] = []  # the same, with lexicon phrases merged
    compound: list[float] = []
    term_counts: Counter[str] = Counter()
    sentiment_sentences: Counter[str] = Counter()  # sentiment -> n sentences
    lexicon_sentences: Counter[tuple[str, str]] = Counter()  # (lexicon, sentiment) -> n
    lexicon_totals: Counter[str] = Counter()  # lexicon -> n sentences, any sentiment
    topic_sentiment_sums: dict[str, float] = defaultdict(float)
    topic_sentence_counts: Counter[str] = Counter()
    # Same two things for sentences that mention *both* lexicons of a pair
    pair_sentiment_sums: dict[tuple[str, str], float] = defaultdict(float)
    pair_sentence_counts: Counter[tuple[str, str]] = Counter()

    for sentence in sentences:
        # --- Word tokenisation (Treebank), lemmatisation, phrase merging ---
        tokens = normalise_tokens(word_tokenize(sentence))
        merged = mwe_tokenizer.tokenize(tokens)
        words.extend(tokens)
        merged_words.extend(merged)

        # --- Sentiment (VADER) ---
        score = sia.polarity_scores(sentence)["compound"]
        compound.append(score)
        sentiment = classify_sentiment(score)
        sentiment_sentences[sentiment] += 1

        # --- Which lexicons does this sentence touch? ---
        # A sentence counts once per lexicon no matter how often it mentions
        # it, but it counts for every lexicon it mentions. So a sentence about
        # "trust in multilateral institutions" is counted for theme_trust and
        # for theme_multilateralism.
        lexicons_here: set[str] = set()
        for token in merged:
            if token in term_to_lexicons:
                term_counts[token] += 1
                lexicons_here.update(term_to_lexicons[token])
        for lexicon in lexicons_here:
            lexicon_sentences[(lexicon, sentiment)] += 1
            lexicon_totals[lexicon] += 1
        for lexicon in SENTIMENT_LEXICONS:
            if lexicon in lexicons_here:
                topic_sentiment_sums[lexicon] += score
                topic_sentence_counts[lexicon] += 1
        for pair in CO_OCCURRENCE_PAIRS:
            if all(lexicon in lexicons_here for lexicon in pair):
                pair_sentiment_sums[pair] += score
                pair_sentence_counts[pair] += 1

    n_sentences = len(sentences)
    n_words = len(words)
    stop_words = english_stopwords()
    n_content_words = sum(word not in stop_words for word in words)

    # Term counts aggregated to one count per lexicon (a term in two lexicons
    # counts for both)
    lexicon_counts: Counter[str] = Counter()
    for term, count in term_counts.items():
        for lexicon in term_to_lexicons[term]:
            lexicon_counts[lexicon] += count

    # --- Clean token stream for bag-of-words / TF-IDF style analyses ---
    removed = clean_stopwords()
    tokens_clean = " ".join(token for token in merged_words if token not in removed)

    result = {
        "text_clean": " ".join(sentences),
        "tokens_clean": tokens_clean,
        "n_sentences": n_sentences,
        "n_words": n_words,
        "n_unique_lemmas": len(set(words)),
        # Share of words that are not stop words (a rough "information density")
        "lexical_density": n_content_words / n_words if n_words else 0.0,
        "mean_sentence_length": n_words / n_sentences if n_sentences else 0.0,
        "sentiment_compound_mean": sum(compound) / n_sentences if n_sentences else 0.0,
        "share_positive_sentences": sentiment_sentences["positive"] / n_sentences if n_sentences else 0.0,
        "share_negative_sentences": sentiment_sentences["negative"] / n_sentences if n_sentences else 0.0,
        "term_counts": dict(term_counts),
    }
    for sentiment in REPORTED_SENTIMENTS:
        result[f"n_{sentiment}_sentences"] = sentiment_sentences[sentiment]
    for lexicon in LEXICONS:
        result[f"{lexicon}_count"] = lexicon_counts[lexicon]
        for sentiment in REPORTED_SENTIMENTS:
            result[f"{lexicon}_{sentiment}_sentences"] = lexicon_sentences[(lexicon, sentiment)]
        # Every sentence mentioning the lexicon, neutral ones included
        result[f"{lexicon}_sentences"] = lexicon_totals[lexicon]
    for lexicon in SENTIMENT_LEXICONS:
        result[lexicon] = int(topic_sentence_counts[lexicon] > 0)
        result[f"{lexicon}_sentiment"] = (
            topic_sentiment_sums[lexicon] / topic_sentence_counts[lexicon]
            if topic_sentence_counts[lexicon]
            else 0.0
        )
    for pair in CO_OCCURRENCE_PAIRS:
        name = "_".join(pair)
        count = pair_sentence_counts[pair]
        result[name] = int(count > 0)
        result[f"{name}_sentences"] = count
        result[f"{name}_sentiment"] = (
            pair_sentiment_sums[pair] / count if count else 0.0
        )
    return result


# ---------------------------------------------------------------------------
# Step 4: build the output datasets
# ---------------------------------------------------------------------------


def sentence_count_columns() -> list[str]:
    """Names of the sentence-count columns, in output order.

    The speech totals first, then one block per lexicon: how many of its
    sentences are positive and negative, and how many mention it at all.
    """
    columns = ["n_sentences"] + [f"n_{s}_sentences" for s in REPORTED_SENTIMENTS]
    for lexicon in LEXICONS:
        columns += [f"{lexicon}_{s}_sentences" for s in REPORTED_SENTIMENTS]
        columns.append(f"{lexicon}_sentences")
    return columns


def build_lexicon_terms_by_year(features: pd.DataFrame, term_counts: pd.Series) -> pd.DataFrame:
    """Long table: how often each lexicon term was used in each year.

    Counts are raw. The corpus grows over time (roughly 50 speeches a year in
    1946, ~190 today), so divide by the year's total word count before
    comparing years::

        words_per_year = features.groupby("year")["n_words"].sum()
    """
    _, term_to_lexicons = lexicon_matcher()
    records = [
        (year, term, count)
        for year, counts in zip(features["year"], term_counts)
        for term, count in counts.items()
    ]
    terms = (
        pd.DataFrame(records, columns=["year", "term", "count"])
        .groupby(["year", "term"], as_index=False)["count"].sum()
    )
    # A term in several lexicons gets one row per lexicon
    terms["lexicon"] = terms["term"].map(term_to_lexicons)
    terms = terms.explode("lexicon")

    terms["term"] = terms["term"].str.replace("_", " ")
    return terms[["year", "lexicon", "term", "count"]].sort_values(
        ["lexicon", "term", "year"]
    )


def merge_country_year(
    features: pd.DataFrame,
    external: pd.DataFrame,
    country_col: str = "Code",
    year_col: str = "Year",
    how: str = "left",
) -> pd.DataFrame:
    """Merge an external country-year dataset onto the speech features.

    The defaults fit Our World in Data CSVs ("Entity", "Code", "Year"). For
    other sources, pass the names of their ISO3 code and year columns, e.g.
    ``merge_country_year(features, happiness, country_col="iso3", year_col="year")``.
    """
    external = external.rename(columns={country_col: "country_code", year_col: "year"})
    external = external.drop(columns=["Entity"], errors="ignore")
    return features.merge(external, on=["country_code", "year"], how=how)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    workers: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run every step and write the datasets. Returns the features table."""
    started = time.perf_counter()
    ensure_nltk_resources()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw speeches and metadata
    speeches = load_speeches(raw_dir / "TXT")
    if limit:  # random sample for fast test runs
        speeches = speeches.sample(n=min(limit, len(speeches)), random_state=42)
    print(f"Loaded {len(speeches):,} speeches")
    countries = load_country_metadata(raw_dir / "unsd_methodology.csv")
    speakers = load_speakers(raw_dir / "Speakers_by_session.xlsx")

    # 2-3. Clean and analyse each speech in parallel (the NLP part is CPU-heavy)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"Running NLP on {workers} worker processes ...")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = zip(speeches["speech"], speeches["session"])
        for i, result in enumerate(executor.map(analyse_speech, jobs, chunksize=16), start=1):
            results.append(result)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(speeches):,} speeches processed")

    nlp = pd.DataFrame(results, index=speeches.index)
    term_counts = nlp.pop("term_counts")
    texts = nlp[["text_clean", "tokens_clean"]]
    nlp = nlp.drop(columns=["text_clean", "tokens_clean"])

    # 4. Attach metadata: country names and regions, then speakers
    keys = speeches[["country_code", "year", "session"]]
    features = (
        pd.concat([keys, nlp], axis=1)
        .merge(countries, on="country_code", how="left")
        .merge(speakers, on=["country_code", "year"], how="left")
    )
    features["speaker_role"] = features["speaker_role"].fillna("unknown")
    features = features[META_COLUMNS + [c for c in features.columns if c not in META_COLUMNS]]

    # Keep the speech index so term counts line up with the rows after sorting
    order = features.assign(_idx=speeches.index).sort_values(["year", "country_code"])
    features = order.drop(columns="_idx").reset_index(drop=True)
    term_counts = term_counts.loc[order["_idx"]].reset_index(drop=True)
    # Stored as JSON so the dict survives the CSV round trip
    features["term_counts"] = term_counts.map(
        lambda counts: json.dumps(
            {term.replace("_", " "): count for term, count in sorted(counts.items())}
        )
    )
    texts = pd.concat([features[["country_code", "year"]], texts.loc[order["_idx"]].reset_index(drop=True)], axis=1)

    # 5. Write the datasets, one table per grain. The word counts and the
    # sentence counts describe the same speeches, so they share a row in
    # speeches_features.csv.
    outputs = {
        "speeches_features.csv": features,
        "speeches_text.csv.gz": texts,
        "lexicon_terms_by_year.csv": build_lexicon_terms_by_year(features, term_counts),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)
        print(f"Wrote {out_dir / filename} ({len(frame):,} rows)")

    # Report speeches that could not be matched to metadata
    missing_region = features.loc[features["region_name"].isna(), "country_code"].unique()
    if len(missing_region):
        print(f"Warning: no region for codes {sorted(missing_region)}")
    print(f"Speeches without speaker info: {features['speaker_name'].isna().sum():,}")
    print(f"Done in {time.perf_counter() - started:.0f}s")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder with the Dataverse files")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where to write the datasets")
    parser.add_argument("--workers", type=int, default=None, help="number of processes (default: CPUs - 1)")
    parser.add_argument("--limit", type=int, default=None, help="only process a random sample of N speeches")
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.out_dir, args.workers, args.limit)


if __name__ == "__main__":
    main()
