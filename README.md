# UN General Debate Analysis

Fundamentals of Data Science — Assignment 1. We analyse the **UN General Debate Corpus
(UNGDC), 1946–2025** (80 sessions, ~11k speeches) against this year's General Assembly
theme, *"Restoring trust, managing transformation"*, and a chosen Sustainable
Development Goal.
Sentiment lexicon — positive/negative words (this can be borrowed from an existing one like Bing)
This repo holds the **preprocessing pipeline**: it turns the raw speech files into three
analysis-ready CSVs, keyed by `(country_code, year)` so they merge cleanly with external
country-year datasets (World Happiness Report, trade data, Our World in Data, …).
See [ASSIGNMENT.md](ASSIGNMENT.md) for the full brief.

The analysis itself is in [notebooks/technology-sdg9.ipynb](notebooks/technology-sdg9.ipynb):
**SDG 9 (Industry, Innovation and Infrastructure)**, asking how the General Debate's
technology talk moved from *transfer* to *access* to *risk* over eighty years, and how
well a speech predicts its country's internet penetration (§8), its connectivity growth
(§9) and its R&D spending (§10).

---

## 1. Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in
`.python-version`).

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/geomimo/UN-General-Debate-Analysis.git
cd UN-General-Debate-Analysis
uv sync          # creates .venv/ and installs everything from uv.lock
```

In VS Code, select `.venv/bin/python` as the interpreter so the notebooks pick up the
same environment.

NLTK models (punkt, stopwords, wordnet, omw-1.4, vader_lexicon) are **not** part of
`uv sync` — the pipeline downloads them automatically on its first run (~50 MB, needs
internet).

## 2. Get the data

`data/` is gitignored (216 MB), so a fresh clone has **no data at all** — you have to
download it before anything runs.

1. **UNGDC corpus** — Harvard Dataverse, [doi:10.7910/DVN/0TJX8Y](https://doi.org/10.7910/DVN/0TJX8Y)
   (use **v14, March 2026**, the version that includes Session 80 / 2025). Download and
   unpack `UNGDC_1946-2025.tar.gz` and `Speakers_by_session.xlsx`. You do *not* need the
   `Raw_PDFs_*.tgz` tarballs.
2. **UNSD M49 country/region table** — export the CSV from
   [unstats.un.org/unsd/methodology/m49](https://unstats.un.org/unsd/methodology/m49/overview/)
   and save it as `unsd_methodology.csv`. It must be the semicolon-separated version with
   the `ISO-alpha3 Code`, `Country or Area`, `Region Name` and `Sub-region Name` columns.

Arrange them exactly like this:

```
data/
├── raw/                          # you download this
│   ├── TXT/
│   │   ├── Session 01 - 1946/
│   │   │   └── ARG_01_1946.txt   # ISO3 _ session _ year
│   │   └── ... (80 session folders)
│   ├── Speakers_by_session.xlsx
│   └── unsd_methodology.csv
└── processed/                    # the pipeline writes this
```

## 3. Run the pipeline

```bash
uv run python -m un_general_debate_analysis.preprocessing               # full corpus
uv run python -m un_general_debate_analysis.preprocessing --limit 300   # quick test run
```

Flags: `--limit N` (random sample of N speeches), `--workers N` (default: CPUs − 1),
`--raw-dir` / `--out-dir`. The NLP is CPU-bound and parallel; a full run takes a few
minutes on a laptop.

⚠️ A `--limit` run **overwrites** the full outputs with a sample. If `data/processed/`
has ~50 rows instead of ~11k, someone ran it with a limit — just re-run without one.

## 4. What you get in `data/processed/`

| File | Grain | What's in it |
| --- | --- | --- |
| `speeches_features.csv` | one row per speech (country-year) | metadata, text stats, VADER sentiment, per-lexicon word **and** sentence counts, `term_counts` — 107 columns |
| `speeches_text.csv.gz` | one row per speech | `text_clean` (readable) and `tokens_clean` (lemmatised, stop-words removed) — for TF-IDF, topic models, word clouds |
| `lexicon_terms_by_year.csv` | year × lexicon × term | raw `count` — shows *which* words drive a trend |

`speeches_features.csv` is the main table. Columns:

- Keys/metadata: `country_code`, `country`, `year`, `session`, `region_name`,
  `sub_region_name`, `speaker_name`, `speaker_post`, `speaker_role`
  (`speaker_role` normalises ~270 free-text job titles into `head_of_state`,
  `head_of_government`, `foreign_minister`, `un_diplomat`, …).
- Text stats: `n_sentences`, `n_words`, `n_unique_lemmas`, `lexical_density`,
  `mean_sentence_length`.
- Sentiment: `sentiment_compound_mean`, `share_positive_sentences`,
  `share_negative_sentences` (VADER, scored per sentence, thresholds ±0.05).
- Per lexicon, a binary presence flag and a mean sentiment: `technology` is 1 when
  the speech has at least one technology sentence, and `technology_sentiment` is the
  mean VADER compound score of those sentences (0.0 when there are none). Same for
  every other lexicon.
- Per pair in `CO_OCCURRENCE_PAIRS`, the same three things for sentences that mention
  **both** — e.g. `technology_frame_peril` (0/1),
  `technology_frame_peril_sentences` (how many) and
  `technology_frame_peril_sentiment`. This is what makes the promise/peril framing
  measurable: a bare count of "threat" is useless, a count of "threat *in a sentence
  that also mentions technology*" is not.
- Per lexicon, a block of four adjacent columns — e.g. for `theme_trust`:
  `theme_trust_count` (how often its terms occur, counted in **words**),
  `theme_trust_positive_sentences`, `theme_trust_negative_sentences` and
  `theme_trust_sentences` (how many **sentences** mention it, in total and by tone).
  So the same table answers "how much does a country talk about trust?" and "does it
  talk about trust in a hopeful or a worried tone?".
- `term_counts`: a **JSON string** of per-term counts — parse it with
  `json.loads`, not `ast.literal_eval`.

Always normalise counts before comparing speeches — speech length varies a lot across
the corpus. Divide word counts by `n_words` and sentence counts by `n_sentences`.

A sentence is counted once per lexicon it mentions, and neutral sentences have no column
of their own (they're the gap between a total and its positive + negative parts). So the
lexicon columns neither sum to `n_sentences` nor are bounded by it.

107 columns is a lot to eyeball. To see just the metadata plus the sentence-level
columns:

```python
from un_general_debate_analysis.preprocessing import META_COLUMNS, sentence_count_columns
df[META_COLUMNS + sentence_count_columns()].head()
```

### Getting the sentences back

`speeches_features.csv` says *how many* sentences mention a lexicon;
[subcorpus.py](src/un_general_debate_analysis/subcorpus.py) gives you the sentences
themselves, labelled with every lexicon they matched and their VADER score. That is
how you check a dictionary instead of trusting it:

```python
from un_general_debate_analysis.subcorpus import lexicon_sentences

texts = pd.read_csv(OUT_DIR / "speeches_text.csv.gz")
tech = lexicon_sentences(texts.query("year >= 1990"), "technology")
tech[tech["frame_peril"] == 1]["sentence"].sample(10)
```

It re-tokenises from scratch, so budget ~5 s per 1,000 speeches.

## 5. Merging in external datasets

`merge_country_year` is set up for Our World in Data CSVs (`Entity`, `Code`, `Year`):

```python
import pandas as pd
from un_general_debate_analysis.preprocessing import merge_country_year

features = pd.read_csv("data/processed/speeches_features.csv")
owid = pd.read_csv("data/external/some-owid-indicator.csv")
df = merge_country_year(features, owid)                       # OWID defaults
df = merge_country_year(features, happiness, country_col="iso3", year_col="year")
```

Both keys are `(country_code, year)` with ISO-3166 alpha-3 codes. Note the corpus also
contains dissolved states (`CSK`, `DDR`, `YUG`, `YMD`) and `EU`, which are hardcoded in
`preprocessing.py` because they're absent from the UNSD table — most external datasets
won't have them either, so expect NaNs there.

### The datasets we actually use

[external.py](src/un_general_debate_analysis/external.py) downloads and caches them, so
nobody has to hunt for a CSV twice. Everything comes back keyed by
`(country_code, year)` with ISO3 codes:

```python
from un_general_debate_analysis.external import load_all, world_bank_indicators, egdi

panel = load_all()                      # all of the below, outer-joined
features.merge(panel, on=["country_code", "year"], how="left")
```

| Function | Source | Coverage |
| --- | --- | --- |
| `world_bank_indicators()` | World Bank WDI API — **connectivity** (internet users %, mobile and broadband per 100, secure servers), **technology investment** (R&D % of GDP = SDG 9.5.1, researchers per million, patents, scientific articles, high-tech and ICT-service exports), and the controls (GDP per capita PPP, population) | 1960–2025, ~240 countries |
| `egdi()` | UN E-Government Development Index (World Bank Data360; reads `data/raw/UN_EGDI_EGDI.csv` if present) | biennial 2003–2024 |
| `world_happiness()` | World Happiness Report `DataForTable2.1.xls` — life ladder, perceived corruption, social support | 2005–2024 |
| `development_groups()` | UNSD M49 — LDC / landlocked / small-island flags | country-level |

Downloads are cached to `data/external/` (gitignored); pass `refresh=True` to re-fetch.
The World Happiness Report is keyed by country *name*, so `external.py` resolves names
to ISO3 against the UNSD table plus a fix-up table; four names stay unmatched
(Hong Kong, Kosovo, Somaliland, Taiwan) and none of them give General Debate speeches.

## 6. Changing what we measure

The topic dictionaries live in [lexicons.py](src/un_general_debate_analysis/lexicons.py).
Twelve lexicons currently, in four groups:

| Group | Lexicons | What it is for |
| --- | --- | --- |
| Theme | `theme_trust`, `theme_transformation`, `theme_multilateralism` | this year's GA theme |
| **SDG 9 facets** | `tech_access`, `tech_frontier`, `tech_security`, `tech_governance`, `tech_economy` | *which* technology a speech is talking about |
| **Framing** | `frame_promise`, `frame_peril` | *how* it is framed — only meaningful inside another lexicon's sentences |
| SDG topics | `technology`, `education` | the umbrella topics |

`technology` is **built as the union of the five `tech_*` facets**, not written out by
hand — see the gotcha below for why that is not optional.

To add or change terms: write them in plain English (`"climate change"`,
`"gender-based violence"`). They go through the same tokenising and lemmatising as the
speeches, so don't pre-lemmatise — `"women"` already matches `"woman"`.

### Three traps in the matching

1. **A phrase hides the single words inside it.** Multi-word terms are merged into one
   token and the longest phrase wins, so once `"digital divide"` is registered anywhere,
   those sentences produce the token `digital_divide` and stop matching a bare
   `"digital"` entry in another lexicon. If a phrase belongs to the umbrella lexicon,
   it has to be listed there too. (This silently cost the old `technology` lexicon
   every mention of `internet`, `cyber` and `digital divide`.)
2. **Tokens are lowercased and stripped of non-letters.** `"AI"` becomes `ai` and is
   then indistinguishable from OCR noise; `"e-commerce"` loses the single-letter `e`
   and matches every mention of *commerce*. `TERM_NOTES` at the bottom of the file
   records each term dropped for this reason and why, so the decision is not
   re-litigated.
3. **Common words need a context.** `"threat"` on its own measures nothing. Add the
   pair to `CO_OCCURRENCE_PAIRS` instead and the pipeline counts sentences that mention
   *both* lexicons — that is how `technology_frame_peril_sentences` is built.

`NEUTRAL_PHRASES` kills the remaining false positives (e.g. `"trust territory"`, the
colonial trusteeship system, is not about *trust*; `"quantum leap"` is not about
quantum computing).

**Re-run the pipeline after editing lexicons** — the counts are baked into the CSVs.

## 7. Layout & working agreements

```
.
├── ASSIGNMENT.md                    # the brief
├── pyproject.toml / uv.lock         # dependencies
├── data/                            # gitignored — see §2
├── notebooks/
│   └── template.ipynb               # copy this to eda-<yourname>.ipynb
└── src/un_general_debate_analysis/  # the installed package
    ├── preprocessing.py             # the pipeline (read its module docstring first)
    ├── lexicons.py                  # keyword dictionaries
    ├── external.py                  # download + cache the external country-year data
    ├── geonames.py                  # blocklists that strip country identity out of the text
    └── subcorpus.py                 # pull the matching sentences back out of the corpus
```

`src/<package>/` is a standard Python src-layout, and `uv sync` installs it into `.venv`.
That's what makes this work from a notebook in *any* directory, with no `sys.path` fiddling:

```python
from un_general_debate_analysis.preprocessing import merge_country_year, sentence_count_columns
```

- **Only importable code goes in `src/un_general_debate_analysis/`.** Notebooks, docs and
  data live outside it.
- **Everyone regenerates `data/` locally.** Processed CSVs are gitignored, so don't count
  on someone else's outputs matching yours — if you change `lexicons.py`, say so, because
  everyone then needs to re-run.
- **One notebook per person.** Start by copying `notebooks/template.ipynb` to
  `notebooks/eda-<yourname>.ipynb` — it loads the tables and sets up the imports.
  `notebooks/technology-sdg9.ipynb` is the SDG 9 analysis (exploratory + predictive). Merging
  two people's edits to the same `.ipynb` is miserable (the diff is JSON with embedded
  outputs), so don't share one. Clear outputs before committing.
- **Get paths from the package, not from `../data/...`.** `OUT_DIR` and `RAW_DIR` are
  absolute, so they work whichever directory your kernel started in.
- Shared, reusable logic belongs in a `.py` module; exploration belongs in your notebook.

## 8. Git workflow

`main` is what everyone clones from, so nobody commits to it directly — you work on a
branch, push it, and open a pull request. Branches keep everyone's notebooks and
lexicon edits from landing on top of each other.

### Start a piece of work

Always branch off an up-to-date `main`:

```bash
git checkout main
git pull                          # fetch + fast-forward main to origin/main
git checkout -b eda-george        # create the branch and switch to it
```

Name branches after what they hold: `eda-<yourname>` for your notebook,
`lexicon-sdg16` or `fix-speaker-roles` for changes to `src/`.

Handy: `git switch <branch>` moves between existing branches, `git branch` lists yours,
`git status` tells you where you are and what's changed.

### Commit

```bash
git status                        # what changed
git diff                          # the actual changes, unstaged
git add notebooks/eda-george.ipynb src/un_general_debate_analysis/lexicons.py
git diff --staged                 # review exactly what you're about to commit
git commit -m "Add SDG16 sentence-level plots"
```

- **Stage files by name, not `git add .`** — `data/` is gitignored, but stray exports,
  `.DS_Store` and scratch scripts are easy to sweep up by accident.
- **Clear notebook outputs before committing** (Kernel → Restart & Clear Output).
  Committed outputs make the JSON diff unreadable and bloat the repo.
- Small commits with a one-line message in the imperative ("Add…", "Fix…", "Rename…")
  are easier to review and to undo than one giant one.

Forgot a file, or want to reword the last message? `git commit --amend` — but only on
commits you haven't pushed yet.

### Push and open a PR

```bash
git push -u origin eda-george     # -u only the first time; afterwards just `git push`
```

Then open the pull request on GitHub (or `gh pr create`). Once it's merged, delete the
branch and start the next piece of work from a fresh `main`.

### Stay in sync: pull and rebase

While you work, other people are merging into `main`. Before you push — and any time you
want their changes — replay your commits on top of the latest `main`:

```bash
git fetch origin                  # update origin/main without touching your files
git rebase origin/main            # move your commits on top of it
```

Rebase gives a straight, readable history instead of merge commits everywhere. The rule
that keeps it safe: **rebase your own unpushed work, never shared `main`.** If you've
already pushed the branch and then rebase it, the push needs
`git push --force-with-lease` (which refuses if someone else pushed to your branch in the
meantime — use it instead of plain `--force`).

To update `main` itself, a plain `git pull` is fine — you never commit there, so it just
fast-forwards.

### When a rebase stops on a conflict

Git pauses and marks the clashing files:

```bash
git status                        # lists "both modified" files
# edit each one, deleting the <<<<<<< ======= >>>>>>> markers
git add <file>
git rebase --continue             # repeat until it finishes
git rebase --abort                # or: back out, nothing changed
```

Conflicting `.ipynb` files are the painful case — the conflict is in JSON, not in your
code. This is why everyone has their own notebook (§7). If it happens anyway, the
quickest fix is usually to keep one side wholesale:

```bash
git checkout --ours notebooks/eda-george.ipynb     # the version already on main
git checkout --theirs notebooks/eda-george.ipynb   # your incoming commit
```

(During a *rebase* "ours" and "theirs" are swapped relative to what you'd expect — ours
is the branch you're replaying onto, theirs is your commit.)

### Getting out of trouble

```bash
git restore <file>                # throw away uncommitted changes to a file
git restore --staged <file>       # unstage, keep the edits
git stash / git stash pop         # park uncommitted work to switch branches
git reset --soft HEAD~1           # undo the last commit, keep the changes staged
git log --oneline --graph --all   # see where every branch actually is
git reflog                        # every HEAD you've been at — recovers "lost" commits
```

Nothing that's been committed is really lost; `git reflog` plus
`git checkout <hash>` gets it back. Avoid `git reset --hard` unless you're sure — that
one *does* discard work.

`data/` is gitignored, so none of this touches your processed CSVs. Switching branches
never makes you re-run the pipeline — unless the branch changes `lexicons.py` or
`preprocessing.py`, in which case re-run it (§3).

## 9. Caveats worth knowing for the write-up

- **Session 80 (2025) is different in kind**: the UN published no validated transcripts,
  so the whole session was transcribed from interpretation audio with Whisper. Expect
  transcription noise, and be careful about reading a 2025 "trend" as real. The pipeline
  strips the presiding officer's intro/outro sentences from that session only.
- **Pre-1994 speaker posts** are inconsistently recorded in the UN data, so
  `speaker_role` is `unknown` for many early speeches.
- Older speeches come from OCR'd PDFs; the cleaner removes page headers, paragraph
  numbers and line-break hyphenation, but artefacts survive. This is what makes short
  acronyms unusable as lexicon terms — see `TERM_NOTES` in `lexicons.py`.
- **A dictionary is not a classifier.** The framing dictionaries were hand-audited on
  40 sentences: `frame_promise` is right about 85% of the time, `frame_peril` about
  65–70% (General Debate speeches list threats, and technology is often one item in
  the list rather than the threat itself). Report a *trend* in these measures, not a
  level, and quote the high-precision `tech_security` facet next to it.
- **VADER does not know this vocabulary.** It scores `cyberattack`, `disinformation`
  and `deepfake` as neutral, so `*_sentiment` columns understate how negative the
  modern technology debate is. That is the reason the framing dictionaries exist.
- **A big coefficient is not a load-bearing feature.** 23 of the 30 strongest word
  features in the §8 connectivity model are country names, which looks like the
  `GroupKFold`-by-country design being defeated by the text itself (98.7% of speeches
  name their own country). It is not: `geonames.py` strips every country name,
  demonym, region, capital and regional body, and out-of-fold R² moves from 0.835 to
  0.834. Place names *on their own* are worth +0.03 R² where the full text is worth
  +0.28. Rank features by coefficient if you like, but measure their contribution by
  deleting them and refitting — §8.1 of the notebook does this at four levels of
  strictness, and §10 repeats it for R&D.
- **Check persistence before claiming a forecast.** R&D intensity has a five-year
  autocorrelation of 0.974, so a model predicting it five years out scores R² 0.47
  without forecasting anything. §10 of the notebook runs the same model at t+0, t+5
  and t+10; the score is flat, which proves it is reading the present. Any
  "we predicted the future" claim against a slow-moving indicator needs that test.
- **The R&D panel is a selected sample.** Only ~107 countries report R&D spending at
  all, and they are richer and more research-active than the rest.

## Citation

> Jankin, S., Baturo, A., & Dasandi, N. (2025). Words to unite nations: The complete
> United Nations General Debate Corpus, 1946–present. *Journal of Peace Research*,
> 62(4), 1339–1351.

> Baturo, A., Dasandi, N., & Mikhaylov, S. (2017). Understanding State Preferences With
> Text As Data: Introducing the UN General Debate Corpus. *Research & Politics*.
