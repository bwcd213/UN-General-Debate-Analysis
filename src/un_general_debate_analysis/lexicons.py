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
    "education": [
        "education", "educational", "school", "schooling", "literacy",
        "illiteracy", "teacher", "student", "pupil", "university",
        "learning", "curriculum", "scholarship", "classroom", "teacher training",
        "vocational training", "skills training", "higher education",
        "primary education", "secondary education", "early childhood education",
        "technical education", "inclusive education", "quality education",
        "distance learning", "online learning", "educational opportunity",
        "school enrollment", "school enrolment", "dropout", "child education",
    ],
    "technology": [
        "technology", "computer", "innovation", "research and development",
        "computation", "digital", "digitalisation", "digitalization",
        "information technology", "communication technology", 
        "information and communication technologies", "ict", "internet access", "broadband",
        "connectivity", "digital infrastructure", "digital economy", "e-commerce",
        "open data", "data science", "big data", "machine learning", "algorithm",
        "cloud computing", "renewable technology", "clean technology",
        "technology transfer", "scientific research", "technological development",
        "research capacity", "innovation ecosystem", "startup", "entrepreneurship",
        "ai", "artificial intelligence", "cloud", "network",
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
