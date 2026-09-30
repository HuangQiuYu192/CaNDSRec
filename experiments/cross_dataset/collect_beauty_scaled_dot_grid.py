#!/usr/bin/env python3
"""Collect a validation-only Beauty scaled-dot temperature screen."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PAIR = re.compile(r"\('([^']+)',\s*([-+0-9.eE]+)\)")
BLOCK = re.compile(r"(best valid result|test result):\s*OrderedDict\(\[(.*?)\]\)", re.S)
NAME = re.compile(r"scaled_dot_tau([0-9.]+)\.log$")


def metrics(path: Path) -> dict[str, dict[str, float]]:
    return {kind: {key: float(value) for key, value in PAIR.findall(body)}
            for kind, body in BLOCK.findall(path.read_text(encoding="utf-8", errors="replace"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.log_root.glob("scaled_dot_tau*.log"):
        match = NAME.search(path.name)
        parsed = metrics(path)
        if not match or "best valid result" not in parsed or "test result" not in parsed:
            continue
        valid, test = parsed["best valid result"], parsed["test result"]
        rows.append({"temperature": float(match.group(1)), "valid_recall@10": valid["recall@10"],
                     "valid_ndcg@10": valid["ndcg@10"], "test_recall@10": test["recall@10"],
                     "test_ndcg@10": test["ndcg@10"], "selected_by_validation": False,
                     "log": str(path)})
    if not rows:
        raise SystemExit("No complete scaled-dot logs found")
    selected = max(rows, key=lambda row: (row["valid_ndcg@10"], -row["temperature"]))
    selected["selected_by_validation"] = True
    rows.sort(key=lambda row: row["temperature"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Beauty scaled-dot temperature screen", "", "Temperature is selected only by validation NDCG@10.", "",
             "| tau | valid Recall@10 | valid NDCG@10 | test Recall@10 | test NDCG@10 | selected |",
             "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['temperature']:g} | {row['valid_recall@10']:.4f} | {row['valid_ndcg@10']:.4f} | {row['test_recall@10']:.4f} | {row['test_ndcg@10']:.4f} | {'yes' if row['selected_by_validation'] else ''} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
