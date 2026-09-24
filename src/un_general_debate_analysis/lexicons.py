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
  and the longest phrase wins. **A phrase therefore hides the single
  words inside it**: once "digital divide" is registered as a phrase,
  those two words become one token and no longer match a bare "digital"
  entry in another lexicon. Anything that should count for the umbrella
  ``technology`` lexicon has to be listed there too — which is why
  ``technology`` is built as the union of the facets below instead of
  being written out by hand.
- A term may belong to several lexicons; it is then counted for each.
- Terms are lowercased and stripped of non-letters before matching, so
  acronyms are indistinguishable from ordinary words ("AI" -> "ai") and
  "e-" prefixes dissolve ("e-commerce" -> "commerce", which then matches
  every mention of trade). Both are avoided below; see TERM_NOTES.
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
        "digitization", "artificial intelligence", "technology",
        "technological", "innovation", "innovative", "innovate",
        "automation", "robotics", "cyber", "cybersecurity", "cyberspace",
        "internet", "biotechnology", "fourth industrial revolution",
        "disruption", "disruptive", "modernization", "modernisation",
    ]
}


# --- SDG 9 (Industry, Innovation and Infrastructure), split into facets ----

TECH_FACETS: dict[str, list[str]] = {
    # Physical and network foundations of digital access (SDG 9.c)
    "digital_infrastructure": [
        "broadband", "connectivity", "digital network", "telecommunications",
        "fibre", "fiber", "fibre optic", "fiber optic", "wireless",
        "satellite", "bandwidth",
         "grid", "submarine cable", "undersea cable",
        "optical fibre", "optical fiber", "rural connectivity",
        "universal access", "affordable access", "last mile",
        "network infrastructure", "digital infrastructure",
        "digital public infrastructure","robotics","semiconductor", "world wide web",
        "web"

    ],

    # Access to internet and digital services
    "internet_and_access": [
        "internet", "online", "internet access", "internet connectivity",
        "digital divide", "digital gap", "digital inclusion",
        "digital literacy", "digital skill",
        "smartphone",
        "cybersecurity", "cloud computing",
        "information technology", "communication technology",
        "information and communication technology",
        "information and communication technologies", "ict",
        "computer", "computing", "computerization", "computerisation",
        "social media","software","mobile phone",'iphone','phone call','mobile aplication',
        "cell phone","artificial intelligence"
    ],
}

# The umbrella technology lexicon is the union of the facets, so that no
# multi-word facet term is silently hidden from it (see the module docstring).
TECHNOLOGY_TERMS: list[str] = sorted(
    {term for terms in TECH_FACETS.values() for term in terms}
)


# --- How technology is framed: promise or peril ---------------------------
#
# These are *not* topic lexicons. They are evaluative vocabularies that are
# only meaningful in combination with another lexicon, via
# CO_OCCURRENCE_PAIRS: a sentence that mentions technology *and* a promise
# word is an opportunity-framed technology sentence. Counting them inside
# technology sentences only is what keeps "threat" from picking up every
# sentence about war.
FRAME_LEXICONS: dict[str, list[str]] = {
    "frame_promise": [
        "opportunity", "potential", "benefit", "beneficial", "promise",
        "promising", "empower", "empowerment", "harness", "unlock",
        "breakthrough", "leapfrog", "prosperity", "enabler", "catalyst",
        "accelerate", "boost", "advance", "progress", "improve",
        "improvement", "solution", "hope", "optimism", "efficiency",
        "opportunity for all", "bridge the gap", "unleash", "flourish",
        "thrive", "achievement", "success", "modernize", "modernise",
        "better life", "quality of life", "well-being",
    ]
}


# --- One lexicon per Sustainable Development Goal --------------------------

# All lexicons the pipeline will count, in output column order.
LEXICONS: dict[str, list[str]] = {
    **THEME_LEXICONS,
    **TECH_FACETS,
    **FRAME_LEXICONS,

}

# Pairs of lexicons whose *same-sentence* overlap is counted as well. Each
# pair adds three columns, e.g. for ("technology", "frame_peril"):
# ``technology_frame_peril`` (0/1), ``technology_frame_peril_sentences``
# and ``technology_frame_peril_sentiment``.
CO_OCCURRENCE_PAIRS: list[tuple[str, str]] = [
    ("digital_infrastructure", "frame_promise"),
    ("internet_and_access", "frame_promise"),
]

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
    "satellite state",  # Cold War bloc politics, not space technology
    "satellite countries",
    "quantum leap",  # a metaphor, not quantum technology
    "clouds of war",
    "network of terrorism",
    "terrorist network",
    "criminal network",
    "network of support",
]

# Terms deliberately left out, and why. Kept here so the decision is not
# re-litigated every time someone reads the lists above.
TERM_NOTES: dict[str, str] = {
    "ai": (
        "Tokens are lowercased, so the acronym 'AI' is indistinguishable "
        "from OCR noise and from French text quoted in speeches. Half of "
        "its corpus hits predate 2010, which is impossible; "
        "'artificial intelligence' is used instead."
    ),
    "network": (
        "Almost always a network of people or crime in this corpus "
        "('terrorist networks', 'networks of solidarity'), not a computer "
        "network. 'neural network' is kept as a phrase."
    ),
    "cloud": (
        "'clouds of war' and 'clouds on the horizon' dominate. "
        "'cloud computing' is kept as a phrase."
    ),
    "e-commerce": (
        "The 'e-' is dropped as a single letter, leaving the phrase to "
        "match every mention of 'commerce'. Written as 'electronic "
        "commerce' instead, which is rarer but not wrong."
    ),
    "e-government": (
        "Same problem: it would collapse to 'government'. "
        "'digital government' is used instead."
    ),
    "quantum": (
        "'quantum leap' is a common rhetorical flourish; only "
        "'quantum computing' and 'quantum technology' are counted."
    ),
    "satellite": (
        "'satellite states' is Cold War vocabulary, not space technology."
    ),
}
