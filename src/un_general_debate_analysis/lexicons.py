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
    ],
    "theme_multilateralism": [
        "multilateralism", "multilateral", "global governance",
        "international cooperation", "solidarity", "international law",
        "charter", "security council", "security council reform",
        "reform of the security council", "reform of the united nations",
    ],
}


# --- SDG 9 (Industry, Innovation and Infrastructure), split into facets ----
#
# Five facets of technology talk. They answer "*which* technology is this
# country talking about?" and are deliberately kept disjoint in meaning,
# although a sentence can of course touch several of them.
TECH_FACETS: dict[str, list[str]] = {
    # 9.c - connectivity and who is left off the network
    "tech_access": [
        "internet", "internet access", "internet connectivity", "broadband",
        "connectivity", "digital divide", "digital gap", "technology gap",
        "digital inclusion", "digital literacy", "digital skill",
        "digital infrastructure", "digital public infrastructure",
        "information technology", "communication technology",
        "information and communication technology",
        "information and communication technologies", "ict",
        "telecommunication", "telecommunications", "telephone",
        "mobile phone", "mobile telephone", "cellular", "fibre optic",
        "fiber optic", "optical fibre", "submarine cable", "undersea cable",
        "rural connectivity", "universal access", "affordable access",
        "digital government", "computer", "computerization",
        "computerisation", "computing",
    ],
    # The emerging-technology vocabulary of each era
    "tech_frontier": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "algorithm", "algorithmic", "big data",
        "data science", "quantum computing", "quantum technology",
        "blockchain", "cryptocurrency", "robotics", "robot", "automation",
        "automated", "biotechnology", "genetic engineering",
        "nanotechnology", "internet of things", "cloud computing",
        "semiconductor", "microchip", "fourth industrial revolution",
        "emerging technology", "frontier technology", "new technology",
        "advanced technology", "generative artificial intelligence",
        "large language model", "autonomous system", "space technology",
        "outer space technology", "drone",
    ],
    # Technology as a threat: the "restoring trust" side of the theme
    "tech_security": [
        "cyber", "cybersecurity", "cyber security", "cyberspace",
        "cyberattack", "cyber attack", "cybercrime", "cyber crime",
        "cyberwarfare", "cyber warfare", "cyber threat", "cyber espionage",
        "hacking", "malware", "ransomware", "data breach", "spyware",
        "digital surveillance", "mass surveillance", "disinformation",
        "misinformation", "fake news", "deepfake", "online hate",
        "hate speech", "digital authoritarianism", "information warfare",
        "lethal autonomous weapon", "autonomous weapon", "killer robot",
        "online exploitation", "child online protection", "data protection",
        "data privacy", "digital privacy",
    ],
    # Who sets the rules, and who gets the technology
    "tech_governance": [
        "technology transfer", "transfer of technology",
        "digital cooperation", "global digital compact", "digital governance",
        "internet governance", "technology governance", "digital regulation",
        "regulation of artificial intelligence", "science and technology",
        "scientific cooperation", "technological cooperation",
        "technical cooperation", "innovation ecosystem", "research capacity",
        "open data", "digital sovereignty", "digital partnership",
        "technology for development", "digital solidarity",
    ],
    # Technology as an economic engine
    "tech_economy": [
        "technology", "technological", "technological development",
        "technological progress", "technological advance", "innovation",
        "innovative", "innovate", "digital", "digitalization",
        "digitalisation", "digitization", "digitisation",
        "digital economy", "digital transformation", "digital technology",
        "digital age", "digital era", "information society",
        "knowledge economy", "knowledge society", "electronic commerce",
        "online commerce", "startup", "entrepreneurship",
        "research and development", "scientific research",
        "clean technology", "green technology", "renewable technology",
        "modernization", "modernisation",
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
    ],
    "frame_peril": [
        "threat", "risk", "danger", "dangerous", "misuse", "abuse",
        "malicious", "weaponization", "weaponisation", "crime", "criminal",
        "attack", "vulnerability", "vulnerable", "exclusion", "excluded",
        "marginalize", "marginalise", "left behind", "widening gap",
        "inequality", "unregulated", "arms race", "harm", "harmful",
        "destabilize", "destabilise", "manipulation", "erosion", "erode",
        "undermine", "exploitation", "concern", "fear", "unchecked",
        "loss of control", "existential", "peril", "dark side", "misused",
        "deepen", "worsen", "widen",
    ],
}


# --- One lexicon per Sustainable Development Goal --------------------------
SDG_LEXICONS: dict[str, list[str]] = {
    # SDG 4 - kept from the earlier education/technology question so the
    # existing notebooks still run; the current analysis focuses on SDG 9.
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
    # SDG 9 - the union of TECH_FACETS above
    "technology": TECHNOLOGY_TERMS,
}

# All lexicons the pipeline will count, in output column order.
LEXICONS: dict[str, list[str]] = {
    **THEME_LEXICONS,
    **TECH_FACETS,
    **FRAME_LEXICONS,
    **SDG_LEXICONS,
}

# Pairs of lexicons whose *same-sentence* overlap is counted as well. Each
# pair adds three columns, e.g. for ("technology", "frame_peril"):
# ``technology_frame_peril`` (0/1), ``technology_frame_peril_sentences``
# and ``technology_frame_peril_sentiment``.
CO_OCCURRENCE_PAIRS: list[tuple[str, str]] = [
    ("technology", "frame_promise"),
    ("technology", "frame_peril"),
    ("technology", "theme_trust"),
    ("technology", "theme_multilateralism"),
    ("technology", "education"),
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
