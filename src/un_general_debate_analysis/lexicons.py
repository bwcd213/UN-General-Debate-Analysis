"""Keyword dictionaries (lexicons) used to measure what speeches talk about.

Each lexicon maps a short name to a list of words or multi-word phrases.
The preprocessing pipeline counts how often each term occurs in a speech,
and how many of the speech's sentences mention the lexicon at all, split
by whether those sentences are positive or negative.

Write terms in plain English (e.g. "climate change", "gender-based
violence"). They go through the same tokenising and lemmatising steps as
the speeches, so "women" also matches "woman" and "transformed" matches
"transform". Nothing needs to be pre-lemmatised by hand.

Notes on matching:
- Multi-word phrases are merged into one token (e.g. "climate_change"),
  and the longest phrase wins. If a phrase contains a single word that
  belongs to another lexicon, the phrase only counts for its own lexicon.
  For example, "technology transfer" counts for SDG 17, not for
  "transformation".
- A term may belong to several lexicons; it is then counted for each.
- These lists are a starting point. Edit them to fit your questions and
  re-run the pipeline.
"""

# --- Lexicons for this year's theme: "Restoring trust, managing transformation" ---
THEME_LEXICONS: dict[str, list[str]] = {
    "theme_trust": [
        "trust", "distrust", "mistrust", "confidence", "confidence-building",
        # "legitimate" is left out: it mostly appears as "legitimate rights"
        "credibility", "credible", "legitimacy",
        "accountability", "accountable", "transparency", "transparent",
        "integrity", "good faith", "reliable", "reliability",
        "disinformation", "misinformation", "polarization", "polarisation",
        "broken promise",
    ],
    "theme_transformation": [
        "transformation", "transform", "transformative", "transition",
        "digital", "digital divide", "digitalization", "digitalisation",
        "digitization", "artificial intelligence", "ai", "technology",
        "technological", "innovation", "innovative", "innovate",
        "automation", "robotics", "cyber", "cybersecurity", "cyberspace",
        "internet", "biotechnology", "fourth industrial revolution",
        "disruption", "disruptive", "modernization", "modernisation",
    ],
    "theme_multilateralism": [
        "multilateralism", "multilateral", "global governance",
        "international cooperation", "solidarity", "international law",
        "charter", "security council", "security council reform",
        "reform of the security council", "reform of the united nations",
    ],
}

# --- One lexicon per Sustainable Development Goal ---
SDG_LEXICONS: dict[str, list[str]] = {
    "sdg04_education": [
        "education", "educational", "school", "schooling", "literacy",
        "illiteracy", "teacher", "student", "pupil", "university",
        "learning", "curriculum", "scholarship",
    ],
    "sdg05_gender_equality": [
        "gender", "gender equality", "women", "girls", "empowerment of women",
        "women's empowerment", "gender-based violence", "sexual violence",
        "genital mutilation", "femicide", "feminist", "equal pay",
        "maternal",
    ],
    "sdg08_decent_work": [
        "employment", "unemployment", "youth unemployment", "job", "worker",
        "labour", "labor", "decent work", "forced labour", "child labour",
        "economic growth", "wage", "entrepreneurship",
    ],
    "sdg09_industry_innovation": [
        "infrastructure", "industrialization", "industrialisation",
        "industrial", "industry", "manufacturing", "connectivity",
        "broadband", "transport", "railway", "innovation",
        "research and development",
    ],
    "sdg16_peace_justice": [
        "peace", "peaceful", "peacebuilding", "peacekeeping", "conflict",
        "armed conflict", "war", "violence", "terrorism", "terrorist",
        "genocide", "justice", "impunity", "rule of law", "human rights",
        "corruption", "institution", "democracy", "democratic", "governance",
        "disarmament", "weapon", "nuclear weapon", "arms race",
        "arms control", "arms trade", "trafficking",
    ],
    "sdg17_partnerships": [
        "partnership", "global partnership", "development assistance",
        "official development assistance", "oda", "aid", "donor", "debt",
        "debt relief", "trade", "investment", "technology transfer",
        "capacity building", "south-south cooperation",
        "triangular cooperation", "financing for development",
    ],
}

# All lexicons the pipeline will count, in output column order.
LEXICONS: dict[str, list[str]] = {**THEME_LEXICONS, **SDG_LEXICONS}

# Phrases that would otherwise produce false matches. They are merged into
# one token but never counted. For example, "trust territory" (the colonial
# trusteeship system, common in 1950s-60s speeches) is not about "trust".
NEUTRAL_PHRASES: list[str] = [
    "trust territory",
    "trust territories",
    "trust fund",
    "vote of confidence",
    "territorial integrity",  # sovereignty, not the "integrity" of institutions
    "nuclear power",  # usually "nuclear powers" (states), not energy
]
