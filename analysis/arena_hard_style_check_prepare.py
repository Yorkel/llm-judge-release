"""Prepare the check of the default setup against Arena-Hard's published 48.6%, scored exactly Arena-Hard's way.

Arena-Hard's leaderboard score (show_result.py, commit 196f6b8) fits one Bradley-Terry model with style control
(length and markdown) over every model's judgments at once. To score our judgments that way, this script takes a
copy of Arena-Hard's published GPT-4.1 judgments for all 27 models (Hugging Face lmarena-ai/arena-hard-auto,
data/arena-hard-v2.0) and, for each of our runs, writes a judge folder in which gpt-4.1-mini's judgments are
replaced by ours. Everything else is Arena-Hard's own data. Then run, in the Arena-Hard directory:
    python show_result.py -j <folder> -f length markdown
Our round 1 shows the baseline model first (Arena-Hard's games[0]); round 2 shows the candidate first (games[1]).

Usage: python arena_hard_style_check_prepare.py <arena-hard directory containing data/arena-hard-v2.0>

This modifies only the nominated external checkout: it creates the ours-* judge folders,
links the other models' judgments and writes/replaces gpt-4.1-mini.jsonl in each folder.
It does not fetch data, call a model API, or execute the external estimator.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS = {
    "ours-may": HERE / "data" / "full_factorial" / "C1F1D1.jsonl",
    "ours-oct-run1": HERE / "data" / "october_rerun" / "run1" / "C1F1D1.jsonl",
    "ours-oct-run2": HERE / "data" / "october_rerun" / "run2" / "C1F1D1.jsonl",
    "ours-oct-run3": HERE / "data" / "october_rerun" / "run3" / "C1F1D1.jsonl",
}


def main():
    parser = argparse.ArgumentParser(
        description=("Prepare derived judge folders in a nominated external Arena-Hard checkout. "
                     "Creates ours-* folders, symlinks other models' judgments, and writes or "
                     "replaces each folder's gpt-4.1-mini.jsonl. Does not run the estimator."),
    )
    parser.add_argument(
        "arena_hard_dir", type=Path,
        help="External checkout containing data/arena-hard-v2.0/model_judgment/gpt-4.1 (will be modified)",
    )
    args = parser.parse_args()
    checkout = args.arena_hard_dir.expanduser().resolve()
    judgments = checkout / "data" / "arena-hard-v2.0" / "model_judgment"
    published = judgments / "gpt-4.1"
    if not published.is_dir():
        parser.error(f"upstream GPT-4.1 judgments not found: {published}")
    for name, source in RUNS.items():
        folder = judgments / name
        folder.mkdir(exist_ok=True)
        for f in published.glob("*.jsonl"):
            link = folder / f.name
            if f.name != "gpt-4.1-mini.jsonl" and not link.exists():
                link.symlink_to(f)
        n = 0
        with (folder / "gpt-4.1-mini.jsonl").open("w") as out:
            for line in source.open():
                r = json.loads(line)
                games = [{"score": r["round1"].get("verdict")}, {"score": r["round2"].get("verdict")}]
                out.write(json.dumps({"uid": r["uid"], "category": "hard_prompt", "judge": "gpt-4.1",
                                      "model": "gpt-4.1-mini", "baseline": "o3-mini-2025-01-31",
                                      "games": games}) + "\n")
                n += 1
        print(f"{name}: {n} judgments from {source.relative_to(HERE)}")


if __name__ == "__main__":
    main()
