"""
collect.py - Download raw GitHub issues for the issue-outcome classification project.

Task: from the text of a GitHub issue, predict how it was closed:
    completed    - the maintainers resolved it
    not_planned  - the maintainers rejected it (won't fix, invalid, out of scope, ...)

What this script does:
    For every repository in REPOS and every month between START_MONTH and END_MONTH,
    it asks the GitHub Search API for up to PER_MONTH issues closed as "completed"
    and up to PER_MONTH issues closed as "not planned" that were CREATED in that month.
    Sampling both classes from the same months means the model cannot tell them apart
    just by dates, version numbers or other time-related clues.

    The issues are saved unmodified (no cleaning) to data/raw/github_issues_raw.jsonl.gz,
    one JSON object per line, gzip-compressed. Cleaning happens in preprocessing/.

Usage:
    1. (Recommended) set a GitHub token so the download is ~3x faster. Never commit it.
           Windows PowerShell:  $env:GITHUB_TOKEN="your_token_here"
           macOS / Linux:       export GITHUB_TOKEN=your_token_here
    2. Quick test (1 repo, 1 month, prints a few issues, saves nothing):
           python data/collect.py --test
    3. Full download (about 10-15 minutes with a token):
           python data/collect.py
"""

import argparse
import calendar
import gzip
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

SEARCH_URL = "https://api.github.com/search/issues"

REPOS = [
    "microsoft/vscode",
    "microsoft/TypeScript",
    "kubernetes/kubernetes",
    "flutter/flutter",
    "home-assistant/core",
    "numpy/numpy",
]

# Our class name -> value of GitHub's "reason:" search qualifier
CLASSES = {
    "completed": "completed",
    "not_planned": '"not planned"',
}

START_MONTH = (2024, 1)   # first month (inclusive) of issue creation dates
END_MONTH = (2025, 12)    # last month (inclusive)
PER_MONTH = 40            # max issues per (repo, month, class)

OUTPUT = Path(__file__).resolve().parent / "raw" / "github_issues_raw.jsonl.gz"


def make_session():
    """Create an HTTP session. Uses GITHUB_TOKEN if it is set."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        delay = 2.1   # search limit with a token: 30 requests per minute
    else:
        print("No GITHUB_TOKEN set: running at the slower unauthenticated speed.\n")
        delay = 6.5   # search limit without a token: 10 requests per minute
    return session, delay


def get(session, params):
    """Call the Search API, waiting out rate limits and retrying temporary errors."""
    for attempt in range(6):
        response = session.get(SEARCH_URL, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (403, 429):
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait = int(retry_after) + 1
            else:
                reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 0) + 2
            print(f"  rate limit reached, waiting {wait:.0f} s ...")
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        sys.exit(f"ERROR {response.status_code}: {response.text[:300]}")
    sys.exit("ERROR: request kept failing after several retries.")


def months(start, end):
    """Yield (first_day, last_day) for every month from start to end, inclusive."""
    year, month = start
    while (year, month) <= end:
        last = calendar.monthrange(year, month)[1]
        yield date(year, month, 1), date(year, month, last)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def search(session, repo, reason, first, last):
    """Return up to PER_MONTH issues of one repo/class created between first and last."""
    query = f"repo:{repo} is:issue is:closed reason:{reason} created:{first}..{last}"
    params = {"q": query, "sort": "created", "order": "desc", "per_page": PER_MONTH}
    return get(session, params)["items"]


def to_record(issue, repo, cls, month, collected_at):
    """Keep the fields we need from a GitHub issue, without changing their content."""
    user = issue.get("user") or {}
    return {
        "repo": repo,
        "number": issue["number"],
        "url": issue["html_url"],
        "title": issue["title"],
        "body": issue["body"],
        "state_reason": issue.get("state_reason"),
        "query_class": cls,
        "query_month": month,
        "labels": [label["name"] for label in issue["labels"]],
        "created_at": issue["created_at"],
        "closed_at": issue["closed_at"],
        "comments": issue["comments"],
        "author_association": issue.get("author_association"),
        "user_login": user.get("login"),
        "user_type": user.get("type"),
        "collected_at": collected_at,
    }


def run_test(session):
    """Fetch one month of one repo and print a few issues, without saving anything."""
    first, last = next(months(START_MONTH, END_MONTH))
    repo = REPOS[0]
    for cls, reason in CLASSES.items():
        items = search(session, repo, reason, first, last)
        print(f"{repo}  {first:%Y-%m}  {cls:12} -> {len(items)} issues")
        for issue in items[:3]:
            print(f"    #{issue['number']}  [{issue.get('state_reason')}]  {issue['title'][:70]}")
    print("\nTest OK. Now run the full download:  python data/collect.py")


def collect(session, delay):
    """Download every (repo, month, class) combination and write them to OUTPUT."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat()
    totals = {}
    with gzip.open(OUTPUT, "wt", encoding="utf-8") as f:
        for repo in REPOS:
            counts = {cls: 0 for cls in CLASSES}
            for first, last in months(START_MONTH, END_MONTH):
                for cls, reason in CLASSES.items():
                    for issue in search(session, repo, reason, first, last):
                        record = to_record(issue, repo, cls, f"{first:%Y-%m}", collected_at)
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts[cls] += 1
                    time.sleep(delay)
            print(f"{repo:24} completed: {counts['completed']:5d}   not_planned: {counts['not_planned']:5d}")
            totals[repo] = counts

    n_completed = sum(c["completed"] for c in totals.values())
    n_rejected = sum(c["not_planned"] for c in totals.values())
    size_mb = OUTPUT.stat().st_size / 1_000_000
    print(f"\nTotal: {n_completed + n_rejected} issues "
          f"({n_completed} completed, {n_rejected} not_planned), {size_mb:.1f} MB")
    print(f"Saved to {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Download raw GitHub issues.")
    parser.add_argument("--test", action="store_true",
                        help="fetch one month of one repo and print it, without saving")
    args = parser.parse_args()

    session, delay = make_session()
    if args.test:
        run_test(session)
    else:
        collect(session, delay)


if __name__ == "__main__":
    main()
