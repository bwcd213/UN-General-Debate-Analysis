"""Download and cache the external country-year datasets.

Everything here returns a tidy frame keyed by ``(country_code, year)`` with
ISO-3166 alpha-3 codes, so it merges straight onto ``speeches_features.csv``
with :func:`preprocessing.merge_country_year` or a plain ``pd.merge``.

Three sources are wired up:

``world_bank_indicators``
    World Development Indicators over the public World Bank API. Connectivity
    (internet users, broadband, mobile), technology investment (R&D spending,
    researchers, patents, scientific articles, high-tech exports) and the
    GDP-per-capita control all come from here.

``egdi``
    The UN E-Government Development Index, read from the World Bank Data360
    export in ``data/raw/`` when it is there and fetched from the Data360 API
    otherwise. Biennial from 2003.

``world_happiness``
    The World Happiness Report panel (``DataForTable2.1.xls``). It is keyed by
    country *name*, so the names are resolved to ISO3 codes against the UNSD
    M49 table plus :data:`WHR_NAME_FIXES`; anything still unmatched is dropped
    and reported by :func:`world_happiness` when ``verbose=True``.

Downloads are cached as CSVs under ``data/external/`` (gitignored, like the
rest of ``data/``). Delete a file there to force a refresh, or pass
``refresh=True``.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from un_general_debate_analysis.preprocessing import RAW_DIR, PROJECT_ROOT

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# The World Bank series this project uses: our column name -> WDI indicator.
WDI_INDICATORS: dict[str, str] = {
    "internet_users_pct": "IT.NET.USER.ZS",      # Individuals using the Internet (% of population)
    "mobile_per_100": "IT.CEL.SETS.P2",          # Mobile cellular subscriptions per 100 people
    "broadband_per_100": "IT.NET.BBND.P2",       # Fixed broadband subscriptions per 100 people
    "gdp_per_capita_ppp": "NY.GDP.PCAP.PP.KD",   # GDP per capita, PPP (constant 2021 international $)
    "rd_spending_pct_gdp": "GB.XPD.RSDV.GD.ZS",  # Research and development expenditure (% of GDP)
    "population": "SP.POP.TOTL",                 # Population, total
    "secure_servers_per_1m": "IT.NET.SECR.P6",   # Secure Internet servers per 1 million people
    # --- investment in the capacity to produce technology (SDG 9.5) ---
    "researchers_per_million": "SP.POP.SCIE.RD.P6",  # Researchers in R&D per million people
    "sci_articles": "IP.JRN.ARTC.SC",            # Scientific and technical journal articles
    "patents_residents": "IP.PAT.RESD",          # Patent applications, residents
    "hightech_exports_pct": "TX.VAL.TECH.MF.ZS", # High-technology exports (% of manufactured exports)
    "ict_service_exports_pct": "BX.GSR.CCIS.ZS", # ICT service exports (% of service exports)
        # --- energy indicators ---
    "energy_use_per_capita": "EG.USE.PCAP.KG.OE",
    "electricity_access_pct": "EG.ELC.ACCS.ZS",
}

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    "?format=json&per_page=25000"
)

DATA360_URL = (
    "https://data360api.worldbank.org/data360/data"
    "?DATABASE_ID=UN_EGDI&INDICATOR=UN_EGDI_EGDI&skip=0"
)

# World Happiness Report country names that the UNSD M49 table spells
# differently. Kosovo, Taiwan and Somaliland have no UNGD speeches, so they
# are simply dropped rather than mapped.
WHR_NAME_FIXES: dict[str, str | None] = {
    "bolivia": "BOL",
    "congo (brazzaville)": "COG",
    "congo (kinshasa)": "COD",
    "czechia": "CZE",
    "czech republic": "CZE",
    "hong kong s.a.r. of china": None,
    "iran": "IRN",
    "ivory coast": "CIV",
    "laos": "LAO",
    "moldova": "MDA",
    "north cyprus": None,
    "northern cyprus": None,
    "palestinian territories": "PSE",
    "russia": "RUS",
    "somaliland region": None,
    "south korea": "KOR",
    "state of palestine": "PSE",
    "syria": "SYR",
    "taiwan province of china": None,
    "tanzania": "TZA",
    "turkiye": "TUR",
    "turkey": "TUR",
    "united kingdom": "GBR",
    "united states": "USA",
    "venezuela": "VEN",
    "vietnam": "VNM",
    "kosovo": None,
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "macedonia": "MKD",
    "north macedonia": "MKD",
    "cape verde": "CPV",
    "gambia": "GMB",
    "cote d'ivoire": "CIV",
}


def _cache_path(name: str) -> Path:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    return EXTERNAL_DIR / f"{name}.csv"


def _fetch_json(url: str, timeout: int = 120) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "un-general-debate-analysis/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def world_bank_indicator(
    name: str, indicator: str | None = None, refresh: bool = False
) -> pd.DataFrame:
    """One World Bank series as ``country_code, year, <name>``.

    ``name`` is a key of :data:`WDI_INDICATORS`, or any column name if the
    WDI indicator code is passed explicitly.
    """
    indicator = indicator or WDI_INDICATORS[name]
    cache = _cache_path(f"worldbank_{indicator}")
    if cache.exists() and not refresh:
        frame = pd.read_csv(cache)
    else:
        payload = _fetch_json(WORLD_BANK_URL.format(indicator=indicator))
        header, rows = payload[0], payload[1]
        if header.get("pages", 1) > 1:  # 25k rows covers every series used here
            raise RuntimeError(f"{indicator}: {header['pages']} pages, raise per_page")
        frame = pd.DataFrame(rows)
        frame = (
            frame.assign(
                country_code=frame["countryiso3code"].astype("string").str.strip(),
                year=frame["date"].astype(int),
            )
            .loc[lambda f: f["country_code"].str.len() == 3]
            .rename(columns={"value": name})[["country_code", "year", name]]
            .dropna(subset=[name])
            .sort_values(["country_code", "year"])
            .reset_index(drop=True)
        )
        frame.to_csv(cache, index=False)
    return frame.rename(columns={frame.columns[-1]: name})


def world_bank_indicators(
    names: list[str] | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Every series in :data:`WDI_INDICATORS`, outer-joined on country-year."""
    names = names or list(WDI_INDICATORS)
    merged: pd.DataFrame | None = None
    for name in names:
        frame = world_bank_indicator(name, refresh=refresh)
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=["country_code", "year"], how="outer")
        )
    return merged.sort_values(["country_code", "year"]).reset_index(drop=True)


def egdi(refresh: bool = False) -> pd.DataFrame:
    """UN E-Government Development Index, ``country_code, year, egdi``.

    Biennial (2003-2024) and 0-1 scaled. Read from the Data360 export in
    ``data/raw/UN_EGDI_EGDI.csv`` if present, otherwise from the API.
    """
    cache = _cache_path("un_egdi")
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    local = RAW_DIR / "UN_EGDI_EGDI.csv"
    if local.exists() and not refresh:
        rows = pd.read_csv(local, low_memory=False)
    else:
        first_page = _fetch_json(DATA360_URL)
        values = first_page["value"]
        total = int(first_page.get("count", len(values)))
        skip = len(values)
        while skip < total:
            page = _fetch_json(DATA360_URL.replace("skip=0", f"skip={skip}"))
            page_values = page["value"]
            values.extend(page_values)
            skip += len(page_values)
        rows = pd.DataFrame(values)

    frame = (
        rows.rename(
            columns={
                "REF_AREA": "country_code",
                "TIME_PERIOD": "year",
                "OBS_VALUE": "egdi",
            }
        )[["country_code", "year", "egdi"]]
        .assign(year=lambda f: f["year"].astype(int), egdi=lambda f: pd.to_numeric(f["egdi"], errors="coerce"))
        # 2003 has a handful of 0.0 placeholders for countries with no survey
        .loc[lambda f: f["egdi"].notna() & (f["egdi"] > 0)]
        .drop_duplicates(["country_code", "year"])
        .sort_values(["country_code", "year"])
        .reset_index(drop=True)
    )
    frame.to_csv(cache, index=False)
    return frame


def _normalise_country_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\s*\(.*?\)\s*$", "", name) if name.endswith(")") else name
    name = name.replace("&", "and").replace(".", "")
    name = re.sub(r"\s+", " ", name)
    return name


def _unsd_name_to_code() -> dict[str, str]:
    """Country name -> ISO3, from the UNSD M49 table used by the pipeline."""
    meta = pd.read_csv(RAW_DIR / "unsd_methodology.csv", sep=";", encoding="utf-8-sig", dtype=str)
    lookup: dict[str, str] = {}
    for name, code in zip(meta["Country or Area"], meta["ISO-alpha3 Code"]):
        if isinstance(name, str) and isinstance(code, str):
            lookup[_normalise_country_name(name)] = code
            # "Bolivia (Plurinational State of)" -> also "bolivia"
            lookup.setdefault(_normalise_country_name(re.sub(r"\s*\(.*?\)", "", name)), code)
    return lookup


def world_happiness(verbose: bool = False) -> pd.DataFrame:
    """World Happiness Report panel, keyed by ``(country_code, year)``.

    Keeps the life evaluation and the two institutional-trust style columns
    that matter for the "restoring trust" theme.
    """
    path = RAW_DIR / "DataForTable2.1.xls"
    raw = pd.read_excel(path)
    lookup = _unsd_name_to_code()

    def resolve(name: str) -> str | None:
        key = _normalise_country_name(name)
        if key in WHR_NAME_FIXES:
            return WHR_NAME_FIXES[key]
        return lookup.get(key)

    frame = raw.rename(
        columns={
            "Country name": "country",
            "year": "year",
            "Life Ladder": "life_ladder",
            "Log GDP per capita": "whr_log_gdp_pc",
            "Social support": "social_support",
            "Freedom to make life choices": "freedom",
            "Perceptions of corruption": "perceived_corruption",
            "Positive affect": "positive_affect",
            "Negative affect": "negative_affect",
        }
    )
    frame["country_code"] = frame["country"].map(resolve)
    unmatched = sorted(frame.loc[frame["country_code"].isna(), "country"].unique())
    if verbose and unmatched:
        print(f"World Happiness: {len(unmatched)} country names unmatched -> {unmatched}")

    columns = [
        "country_code", "year", "life_ladder", "whr_log_gdp_pc", "social_support",
        "freedom", "perceived_corruption", "positive_affect", "negative_affect",
    ]
    return (
        frame.dropna(subset=["country_code"])[columns]
        .astype({"year": int})
        .sort_values(["country_code", "year"])
        .reset_index(drop=True)
    )


def development_groups() -> pd.DataFrame:
    """UNSD development-group flags: LDC, landlocked and small-island states."""
    meta = pd.read_csv(RAW_DIR / "unsd_methodology.csv", sep=";", encoding="utf-8-sig", dtype=str)
    flags = meta.rename(
        columns={
            "ISO-alpha3 Code": "country_code",
            "Least Developed Countries (LDC)": "ldc",
            "Land Locked Developing Countries (LLDC)": "lldc",
            "Small Island Developing States (SIDS)": "sids",
        }
    )[["country_code", "ldc", "lldc", "sids"]]
    for column in ["ldc", "lldc", "sids"]:
        flags[column] = flags[column].notna().astype(int)
    return flags.dropna(subset=["country_code"]).reset_index(drop=True)


def load_all(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Every external source, outer-joined into one country-year panel."""
    panel = world_bank_indicators(refresh=refresh)
    panel = panel.merge(egdi(refresh=refresh), on=["country_code", "year"], how="outer")
    panel = panel.merge(world_happiness(verbose=verbose), on=["country_code", "year"], how="outer")
    panel = panel.merge(development_groups(), on="country_code", how="left")
    panel = panel.sort_values(["country_code", "year"]).reset_index(drop=True)
    if verbose:
        rows, countries = len(panel), panel["country_code"].nunique()
        print(f"external panel: {rows:,} country-years, {countries} countries, "
              f"{panel['year'].min()}-{panel['year'].max()}")
    return panel


if __name__ == "__main__":  # `uv run python -m un_general_debate_analysis.external`
    print(load_all(verbose=True).head())
