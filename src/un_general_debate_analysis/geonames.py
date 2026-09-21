"""Strip country identity out of the token stream before a text model sees it.

Why this module exists
----------------------
The predictive sections fit TF-IDF models on ``tokens_clean`` and validate
them with ``GroupKFold`` by country, so that the test question is *"here is a
country you have never read"*. That design is defeated by the speeches
themselves: 93% of them name their own country, and the corpus names every
other country too. The model can therefore learn "Iceland is a rich, wired
place" from the *other* 5,300 speeches that mention Iceland, and recognise
Iceland's own held-out speech by its name. Inspecting the ridge coefficients
makes it plain - ``iran``, ``iceland``, ``monaco``, ``malagasy``,
``turkmenistan`` and ``african`` are among the strongest features.

That is a country lookup table wearing a text model's clothes. This module
builds the blocklists that remove it, at three levels of strictness:

``self``
    only the speaking country's own names and demonyms. Answers "is the model
    just reading the letterhead?"
``country``
    every country name and demonym in the world, from any speaker.
``geo``
    ``country`` plus regions, geographic adjectives, capitals and regional
    organisations - everything that says *where* a speech is from.

Names come from ``data/raw/unsd_methodology.csv`` (the UNSD M49 list the rest
of the pipeline uses for regions), expanded with the regular demonym suffixes
and topped up with a hand-written list of the irregular ones, which no rule
produces: *French*, *Dutch*, *Swiss*, *Malagasy*, *Burkinabe*.

    from un_general_debate_analysis.geonames import scrub_tokens, country_blocklist

    clean = [scrub_tokens(t, country_blocklist()) for t in df["tokens_clean"]]
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from .preprocessing import RAW_DIR

__all__ = [
    "country_blocklist",
    "geo_blocklist",
    "own_country_blocklist",
    "scrub_tokens",
    "scrub_series",
]


# ---------------------------------------------------------------------------
# What a "country token" is not
# ---------------------------------------------------------------------------

# Tokens that appear inside official country names but are ordinary English
# words. "Holy See" would otherwise block "see", "Isle of Man" would block
# "man" (7,505 uses, nearly all of them about humanity) and "Great Britain"
# would block "great" (34,123 uses). The distinguishing token of each of
# those countries - "holy", "britain" - is kept.
GENERIC_NAME_WORDS = frozenset(
    """
    administrative and central christmas cook democratic eastern federation great
    heard island islands islamic jan jersey keeling kingdom man mcdonald middle
    minor new north northern ocean outlying part people plurinational region
    republic saint sandwich see south southern special state states sub territory
    territories the union united western bolivarian helena virgin
    """.split()
)

# Forms the suffix rules generate that are not geographic at all.
DERIVED_FALSE_POSITIVES = frozenset(
    """
    andi cooker easterner greater mani mann martini newer norther northerner
    parti partite republican saner sani sann seen sinter southerner srian staten
    thei verdi westerner
    """.split()
)

# Demonyms and adjectives no suffix rule produces from the country name.
IRREGULAR_DEMONYMS = frozenset(
    """
    afghan basotho batswana beninois bissauguinean britisher burkinabe burmese
    cabo cape catalan chadien comorian congolese cypriot czech danish
    dane danes dprk dutch emirati emirates english finn finnish flemish french
    german greek herzegovinian hellenic icelandic indonesians irish italians
    ivorian ivoirian kanak kazakh kiwi kyrgyz lao laotian lusophone magyar
    malagasy maldivian manx mauritian montenegrin motswana mozambican nigerien
    norwegian persian pole polish portuguese quebecois rok russia russian
    sammarinese saudi scot scots scottish seychellois sinhalese slovak slovene
    spaniard spanish swazi swede swedish swiss tamil thai tibetan tswana turk
    turkic turkish turks turkiye ukrainians uzbek welsh yugoslav zairean zairian
    """.split()
)

# Countries and territories the M49 list does not carry because they no longer
# exist, or carries under a different name than the speeches use.
HISTORICAL_AND_ALIAS_NAMES = frozenset(
    """
    abyssinia biafra burma byelorussia byelorussian ceylon czechoslovak
    czechoslovakia dahomey formosa gaza holland kampuchea katanga kashmir
    kosovo macedonia nagorno karabakh palestinian persia rhodesia sahrawi
    siam soviet swaziland taiwan taiwanese tanganyika tibet transnistria
    ussr volta vietnam vietnamese yugoslavia zaire zanzibar
    britain britons dprk prc syria somali darussalam brunei eire
    czechoslovakian myanma niue oman
    """.split()
)

# Level 3 only: where a speech is from, without naming the country.
REGION_WORDS = frozenset(
    """
    africa african america americas american andean anglophone antilles arab
    arabic asia asian atlantic balkan balkans baltic benelux caribbean caucasus
    caucasian europe european francophone gulf hemisphere himalaya himalayan
    horn iberian latin levant maghreb mashreq mediterranean melanesia melanesian
    mesoamerica micronesian nordic occidental oceania oceanian oriental
    pacific patagonia polynesia polynesian sahara saharan sahel sahelian
    scandinavia scandinavian subsaharan transatlantic east eastern west
    western arctic antarctic continent continental kingdom principality
    sultanate emirate emirates archipelago isthmus peninsula subcontinent
    """.split()
)

# Level 3 only: regional organisations are a country lookup by another name -
# "ecowas" locates a speech in West Africa as precisely as "Senegal" does.
REGIONAL_ORGANISATIONS = frozenset(
    """
    alba andean asean au caricom cis comesa csto eac ecowas eu eurasian
    gcc igad mercosur nato oau oas oecs oic opec osce sadc saarc sica
    unasur visegrad
    """.split()
)

# Level 3 only: capitals and cities that identify a speaker just as well.
CAPITAL_CITIES = frozenset(
    """
    abidjan abuja accra addis ababa algiers amman ankara antananarivo asuncion
    athens baghdad baku bamako bangkok beijing beirut belgrade berlin bern
    bishkek bissau bogota brasilia bratislava brazzaville brussels bucharest
    budapest buenos aires cairo canberra caracas chisinau colombo conakry
    copenhagen dakar damascus dhaka doha dublin dushanbe freetown gaborone
    georgetown hanoi harare havana helsinki islamabad jakarta jerusalem kabul
    kampala kathmandu khartoum kiev kyiv kigali kingston kinshasa kuala lumpur
    kuwait lagos lima lisbon ljubljana lome london luanda lusaka madrid managua
    manila maputo maseru mexico minsk mogadishu monrovia montevideo moscow
    nairobi nassau niamey nicosia nouakchott ouagadougou panama paramaribo
    paris phnom prague pretoria pyongyang quito rabat riga riyadh
    rome sanaa santiago seoul singapore skopje sofia stockholm suva taipei
    tallinn tashkent tbilisi tegucigalpa tehran tirana tokyo tripoli tunis
    ulaanbaatar vientiane vilnius warsaw windhoek yaounde yerevan zagreb
    """.split()
)

# Suffixes that turn a country name into its demonym. Applied to the name as
# it stands, and to the name minus a final vowel ("Kenya" -> "Kenyan",
# "Rwanda" -> "Rwandese", "Iraq" -> "Iraqi").
_SUFFIXES_FULL = ("", "n", "an", "ian", "ese", "ish", "i", "ite", "er", "ans", "ians")
_SUFFIXES_STEM = ("an", "ian", "ese", "ish", "i", "ite", "ans", "ians")

_M49_COLUMNS = {
    "country": "Country or Area",
    "region": "Region Name",
    "sub_region": "Sub-region Name",
    "intermediate": "Intermediate Region Name",
}


def _name_tokens(name: str) -> set[str]:
    """Lowercase word tokens of a country name, minus the generic ones."""
    return {
        word
        for word in re.findall(r"[^\W\d_]+", name.lower())
        if len(word) > 2 and word not in GENERIC_NAME_WORDS
    }


def _expand(tokens: set[str]) -> set[str]:
    """Add the regular demonym forms of every name token."""
    forms = set()
    for token in tokens:
        forms.update(token + suffix for suffix in _SUFFIXES_FULL)
        if token[-1] in "aeouy" and len(token) >= 5:
            stem = token[:-1]
            forms.update(stem + suffix for suffix in _SUFFIXES_STEM)
    return forms - DERIVED_FALSE_POSITIVES


@lru_cache(maxsize=1)
def _m49() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "unsd_methodology.csv", sep=";",
                       encoding="utf-8-sig", dtype=str)


@lru_cache(maxsize=1)
def country_blocklist() -> frozenset[str]:
    """Every country name and demonym, from any speaker (level 2)."""
    names: set[str] = set()
    for name in _m49()[_M49_COLUMNS["country"]].dropna():
        names |= _name_tokens(name)
    return frozenset(
        _expand(names) | IRREGULAR_DEMONYMS | _expand(HISTORICAL_AND_ALIAS_NAMES)
    )


@lru_cache(maxsize=1)
def geo_blocklist() -> frozenset[str]:
    """Country names plus regions, capitals and regional bodies (level 3)."""
    regions: set[str] = set()
    for column in ("region", "sub_region", "intermediate"):
        for name in _m49()[_M49_COLUMNS[column]].dropna():
            regions |= _name_tokens(name)
    return frozenset(
        country_blocklist()
        | _expand(regions)
        | REGION_WORDS
        | REGIONAL_ORGANISATIONS
        | CAPITAL_CITIES
    )


@lru_cache(maxsize=1)
def own_country_blocklist() -> dict[str, frozenset[str]]:
    """``country_code`` -> that country's own names and demonyms (level 1).

    Keyed by ISO-alpha3 so it can be looked up per speech. The corpus also
    holds four states that no longer exist and the European Union, which the
    M49 list does not carry; they get their names here.
    """
    lookup: dict[str, frozenset[str]] = {}
    m49 = _m49()
    for code, name in zip(m49["ISO-alpha3 Code"], m49[_M49_COLUMNS["country"]]):
        if pd.isna(code) or pd.isna(name):
            continue
        lookup[code] = frozenset(_expand(_name_tokens(name)) - DERIVED_FALSE_POSITIVES)

    extra = {
        "CSK": "Czechoslovakia Czechoslovak Czech Slovak",
        "DDR": "German Germany Democratic Republic",
        "YUG": "Yugoslavia Yugoslav Serbia Serbian",
        "YMD": "Yemen Yemeni Democratic",
        "EU": "Europe European Union",
        # The M49 name is not the one the speeches use.
        "GBR": "Britain British United Kingdom England English Scotland Wales",
        "USA": "America American States United",
        "RUS": "Russia Russian Federation Soviet USSR",
        "TUR": "Turkiye Turkey Turkish Turk Turks",
        "VNM": "Viet Nam Vietnam Vietnamese",
        "IRN": "Iran Iranian Persia Persian",
        "MMR": "Myanmar Burma Burmese",
        "PRK": "Korea Korean DPRK",
        "KOR": "Korea Korean",
        "CIV": "Cote Ivoire Ivorian Ivoirian Ivory Coast",
        "NLD": "Netherlands Dutch Holland",
        "CHE": "Switzerland Swiss",
        "MDG": "Madagascar Malagasy",
        "BFA": "Burkina Faso Burkinabe Volta",
        "COD": "Congo Congolese Zaire Zairian Democratic",
        "SWZ": "Eswatini Swaziland Swazi",
        "CPV": "Cabo Verde Cape Verdean",
        "MKD": "Macedonia Macedonian",
        "LKA": "Sri Lanka Lankan Ceylon Sinhalese",
        "GRC": "Greece Greek Hellenic",
        "ESP": "Spain Spanish Spaniard",
        "PRT": "Portugal Portuguese",
        "FRA": "France French",
        "DEU": "Germany German",
        "DNK": "Denmark Danish Dane",
        "SWE": "Sweden Swedish Swede",
        "NOR": "Norway Norwegian",
        "FIN": "Finland Finnish Finn",
        "POL": "Poland Polish Pole",
        "CZE": "Czechia Czech Czechoslovakia",
        "SVK": "Slovakia Slovak Czechoslovakia",
        "HUN": "Hungary Hungarian Magyar",
        "ISL": "Iceland Icelandic Icelander",
        "IRL": "Ireland Irish",
        "THA": "Thailand Thai Siam",
        "LAO": "Lao Laos Laotian",
        "PHL": "Philippines Filipino Philippine",
        "SAU": "Saudi Arabia Arabian",
        "ARE": "Emirates Emirati Arab",
        "KHM": "Cambodia Cambodian Kampuchea",
        "BEN": "Benin Beninese Dahomey",
        "TZA": "Tanzania Tanzanian Tanganyika Zanzibar",
        "ZWE": "Zimbabwe Zimbabwean Rhodesia",
        "BLR": "Belarus Belarusian Byelorussia",
        "ETH": "Ethiopia Ethiopian Abyssinia",
        "PSE": "Palestine Palestinian Gaza",
        "ISR": "Israel Israeli",
    }
    for code, blob in extra.items():
        lookup[code] = frozenset(
            lookup.get(code, frozenset()) | _expand(_name_tokens(blob))
        )
    return lookup


def scrub_tokens(text: str, blocklist: frozenset[str]) -> str:
    """Drop every blocklisted token from a whitespace-joined token stream.

    ``tokens_clean`` is already lowercased and lemmatised, so a set membership
    test is all that is needed. Multi-word lexicon phrases are joined with an
    underscore ("digital_divide") and are left alone - none of them is a place
    name. Dropping the unigrams also removes the bigrams TF-IDF would have
    built from them ("north korea", "sub saharan").
    """
    if not text:
        return ""
    return " ".join(token for token in text.split() if token not in blocklist)


def scrub_series(
    texts: pd.Series, blocklist: frozenset[str], codes: pd.Series | None = None
) -> pd.Series:
    """Scrub a column of token streams.

    With ``codes`` (a matching column of ISO-alpha3 codes) the blocklist is
    taken per row from :func:`own_country_blocklist` and ``blocklist`` is
    added to every row, which is how the "own country only" level is built.
    """
    texts = texts.fillna("")
    if codes is None:
        return texts.map(lambda t: scrub_tokens(t, blocklist))
    own = own_country_blocklist()
    empty: frozenset[str] = frozenset()
    return pd.Series(
        [
            scrub_tokens(text, blocklist | own.get(code, empty))
            for text, code in zip(texts, codes)
        ],
        index=texts.index,
    )
