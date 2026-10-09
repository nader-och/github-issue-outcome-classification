# Contribution – Nader Ochen

- **Student ID:** 2309116429
- **GitHub:** [nader-och](https://github.com/nader-och)
- **Role:** Data collection and repository setup

## Summary

I defined the project task, set up the repository, designed the data sampling strategy, wrote the data collection script, and collected the raw dataset (10,399 GitHub issues) that every later step of the pipeline uses.

## What I did

### 1. Task selection
- Proposed the task: predicting from an issue's title and body whether it will be closed as **completed** or **not planned**.
- Checked it against the tasks listed in the course slides to make sure it is not sentiment, spam, topic or one of the other listed problems.
- Confirmed the topic with the instructor before starting.

### 2. Repository setup
- Created the public repository with an MIT `LICENSE`, a Python `.gitignore` (extended to exclude `.vscode/`) and `requirements.txt`.

### 3. Sampling design
- Selected 6 large, active open-source repositories: `microsoft/vscode`, `microsoft/TypeScript`, `kubernetes/kubernetes`, `flutter/flutter`, `home-assistant/core`, `numpy/numpy`.
- Restricted the data to issues **created between January 2024 and December 2025**.
- Sampled **both classes from the same months**: for every repository and every month, up to 40 issues closed as `completed` and up to 40 closed as `not_planned`. This prevents the model from separating the classes using dates, version numbers or other time-related clues.

### 4. Collection script (`data/collect.py`)
- Queries the GitHub Search API using the `reason:completed` / `reason:"not planned"` qualifiers, so labels come from how maintainers actually closed each issue (no manual labelling).
- Handles the API rate limit (30 requests/min with a token) and retries automatically on network failures and server errors.
- Writes to a temporary `.partial` file and renames it only when the download completes, so an interrupted run can never leave an incomplete dataset under the real file name.
- Includes a `--test` mode that checks one repository and one month without saving anything.
- Saves the issues **unmodified** (cleaning is a separate, documented step) as gzip-compressed JSON Lines.

### 5. Raw dataset (`data/raw/github_issues_raw.jsonl.gz`)

| Repository | completed | not_planned |
|---|---:|---:|
| microsoft/vscode | 960 | 960 |
| microsoft/TypeScript | 844 | 936 |
| kubernetes/kubernetes | 960 | 748 |
| flutter/flutter | 960 | 957 |
| home-assistant/core | 960 | 960 |
| numpy/numpy | 878 | 276 |
| **Total: 10,399** | **5,562 (53.5%)** | **4,837 (46.5%)** |

### 6. Leakage rules for the rest of the pipeline
- Defined that models may only use the issue **title and body**. Fields such as labels, comment counts, closing date and author association are set or change after triage, so using them would leak the answer.

### 7. Documentation
- Wrote the **Dataset** section of the main `README.md` (source, sampling, counts, fields, reproduction steps, license note).

## Commits
- Add data folder and `requirements.txt`
- Add raw dataset (10,399 issues) and robust collection script
