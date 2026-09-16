# UN General Debate Analysis

Fundamentals of Data Science — Assignment 1. We analyse the **UN General Debate Corpus
(UNGDC), 1946–2025** (80 sessions, ~11k speeches) against this year's General Assembly
theme, *"Restoring trust, managing transformation"*, and a chosen Sustainable
Development Goal.

This repo holds the **preprocessing pipeline**: it turns the raw speech files into three
analysis-ready CSVs, keyed by `(country_code, year)` so they merge cleanly with external
country-year datasets (World Happiness Report, trade data, Our World in Data, …).
See [ASSIGNMENT.md](ASSIGNMENT.md) for the full brief.

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
| `speeches_features.csv` | one row per speech (country-year) | metadata, text stats, VADER sentiment, per-lexicon word **and** sentence counts, `term_counts` — 56 columns |
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

56 columns is a lot to eyeball. To see just the metadata plus the sentence-level columns:

```python
from un_general_debate_analysis.preprocessing import META_COLUMNS, sentence_count_columns
df[META_COLUMNS + sentence_count_columns()].head()   # 39 columns instead of 56
```

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

## 6. Changing what we measure

The topic dictionaries live in [lexicons.py](src/un_general_debate_analysis/lexicons.py).
Nine lexicons currently: `theme_trust`, `theme_transformation`, `theme_multilateralism`,
and `sdg04_education`, `sdg05_gender_equality`, `sdg08_decent_work`,
`sdg09_industry_innovation`, `sdg16_peace_justice`, `sdg17_partnerships`.

To add or change terms: write them in plain English (`"climate change"`,
`"gender-based violence"`). They go through the same tokenising and lemmatising as the
speeches, so don't pre-lemmatise — `"women"` already matches `"woman"`. Multi-word terms
are merged into a single token and the longest phrase wins. `NEUTRAL_PHRASES` at the
bottom of the file kills false positives (e.g. `"trust territory"`, the colonial
trusteeship system, is not about *trust*).

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
    └── lexicons.py                  # keyword dictionaries
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
  `notebooks/eda-<yourname>.ipynb` — it loads the tables and sets up the imports. Merging
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
  numbers and line-break hyphenation, but artefacts survive.

## Citation

> Jankin, S., Baturo, A., & Dasandi, N. (2025). Words to unite nations: The complete
> United Nations General Debate Corpus, 1946–present. *Journal of Peace Research*,
> 62(4), 1339–1351.

> Baturo, A., Dasandi, N., & Mikhaylov, S. (2017). Understanding State Preferences With
> Text As Data: Introducing the UN General Debate Corpus. *Research & Politics*.
