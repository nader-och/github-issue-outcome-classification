# Contribution – Nader Ochen

- **Student ID:** 2309116429
- **GitHub:** [nader-och](https://github.com/nader-och)
- **Roles:** Data collection, preprocessing and repository setup (Roles 1 and 2)

## How the team worked

The three of us (Nader Ochen, Çınar Sabancı, Ahmed Rakha) worked together in joint sessions on my computer. For that reason, **every commit in the repository was pushed from my GitHub account**, including the parts my teammates were responsible for. Each member's own file describes the part they were responsible for.

## Summary

I defined the project task, set up the repository, designed the sampling strategy, wrote the data collection script and collected the raw dataset (10,399 GitHub issues). I then wrote the cleaning and splitting pipeline that turns it into the leak-free train / validation / test sets used by every model.

## Role 1: Data collection and repository setup

### Task selection
- Proposed the task: predicting from an issue's title and body whether it will be closed as **completed** or **not planned**.
- Checked it against the tasks listed in the course slides to make sure it is not sentiment, spam, topic or one of the other listed problems.
- Confirmed the topic with the instructor before starting.

### Repository setup
- Created the public repository with an MIT `LICENSE`, a Python `.gitignore` (extended to exclude `.vscode/`) and `requirements.txt`.

### Sampling design
- Selected 6 large, active open-source repositories: `microsoft/vscode`, `microsoft/TypeScript`, `kubernetes/kubernetes`, `flutter/flutter`, `home-assistant/core`, `numpy/numpy`.
- Restricted the data to issues **created between January 2024 and December 2025**.
- Sampled **both classes from the same months**: for every repository and every month, up to 40 issues closed as `completed` and up to 40 closed as `not_planned`. This prevents the model from separating the classes using dates, version numbers or other time-related clues.

### Collection script (`data/collect.py`)
- Queries the GitHub Search API using the `reason:completed` / `reason:"not planned"` qualifiers, so labels come from how maintainers actually closed each issue (no manual labelling).
- Handles the API rate limit (30 requests/min with a token) and retries automatically on network failures and server errors.
- Writes to a temporary `.partial` file and renames it only when the download completes, so an interrupted run can never leave an incomplete dataset under the real file name.
- Includes a `--test` mode that checks one repository and one month without saving anything.
- Saves the issues **unmodified** (cleaning is a separate, documented step) as gzip-compressed JSON Lines.

### Raw dataset (`data/raw/github_issues_raw.jsonl.gz`)

| Repository | completed | not_planned |
|---|---:|---:|
| microsoft/vscode | 960 | 960 |
| microsoft/TypeScript | 844 | 936 |
| kubernetes/kubernetes | 960 | 748 |
| flutter/flutter | 960 | 957 |
| home-assistant/core | 960 | 960 |
| numpy/numpy | 878 | 276 |
| **Total: 10,399** | **5,562 (53.5%)** | **4,837 (46.5%)** |

### Leakage rules for the rest of the pipeline
- Defined that models may only use the issue **title and body**. Fields such as labels, comment counts, closing date and author association are set or change after triage, so using them would leak the answer.

## Role 2: Preprocessing (`preprocessing/clean.py`)

- **Label consistency:** keeps only issues whose close reason at download time still matches the class they were collected for.
- **Bot removal:** removes issues written by `Bot` accounts, plus two CI accounts registered as normal users (`typescript-bot`, `fluttergithubbot`) that I found by spot-checking the most frequent authors of each repository. Their template reports have a strongly skewed label and would let a model score by memorising them.
- **Text cleaning:** builds the input from title + body only, removes template comments and unanswered template fields, and replaces code blocks, links, images, collapsed `<details>` blocks and long stack traces with `[CODE]`, `[URL]`, `[IMAGE]`, `[DETAILS]` and `[TRACE]` tokens, so the information survives while the text stays short enough for transformers.
- **Duplicate removal before splitting:** exact duplicate texts, and titles within a repository that differ only in numbers (e.g. templated Flutter infrastructure tickets), so the same text can never appear in both train and test.
- **Split:** 80 / 10 / 10, stratified on repository + label together (`random_state = 42`), with built-in checks that no `id` or text is shared between splits and that every split has both classes for every repository.
- **Reproducibility:** stable sorting and fixed line endings make reruns byte-identical; `cleaning_report.json` records the counts after every step and the SHA-256 of each CSV.
- **Result:** 10,399 raw → 9,761 clean issues (train 7,808 / val 976 / test 977).

## Documentation
- Wrote the **Task**, **Dataset** and **Preprocessing** sections of the main `README.md` and the full `preprocessing/README.md`.

## Commits (all pushed from my account)

| Commit | Description | Part |
|---|---|---|
| `302653c` | Add data folder and `requirements.txt` | Role 1 |
| `e8a1bb0` | Add raw dataset (10,399 issues) and robust collection script | Role 1 |
| `6b07648` | Add dataset documentation and contribution file | Role 1 |
| `34c65d3` | Add cleaning and splitting script with cleaned train/val/test data | Role 2 |
| `53aa676` | Remove course slides (repository cleanup) | Setup |

Commits `e072779`, `08e326b` (Role 3) and `a66cdc7`, `7ac6f1a`, `8b2a29c` (Role 4) were also pushed from my account during our joint sessions; they are described in Çınar's and Ahmed's files.
