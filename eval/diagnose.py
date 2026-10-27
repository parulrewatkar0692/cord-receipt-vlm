#!/usr/bin/env python3
"""Why does a run score what it scores? Reads a predictions file from the probe notebook.

Reports, for the validation receipts:
  - answers that are not JSON, and how many of them look cut off (no closing brace)
  - how often the model writes numbers instead of the printed text (16500 for "16,500")
  - field F1 again with every value reduced to its digits for price and count fields, which
    shows how much of the gap is formatting rather than reading

Usage
  python eval/diagnose.py /content/cord_validation_qwen25vl3b_zeroshot.jsonl
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import doc_scores, load_labels, micro, parse  # noqa: E402

NUMERIC = re.compile(r"(price|cnt|unitprice)$")


def digits_only(obj):
    """Copy of a parse with price and count values reduced to their digits."""
    if isinstance(obj, dict):
        return {k: (re.sub(r"\D", "", str(v)) if NUMERIC.search(k) and not isinstance(v, (dict, list))
                    else digits_only(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [digits_only(v) for v in obj]
    return obj


def main(path):
    preds = {p["id"]: p["output"] for p in map(json.loads, open(path, encoding="utf-8"))}
    rows = [r for r in load_labels("validation") if r["id"] in preds]
    parsed = {r["id"]: parse(preds[r["id"]])[0] for r in rows}

    bad = [r["id"] for r in rows if parsed[r["id"]] is None]
    cut = [i for i in bad if preds[i].rstrip().rstrip("`").rstrip()[-1:] != "}"]
    print(f"not JSON: {len(bad)} of {len(rows)}; {len(cut)} of them look cut off (no closing brace)")
    for i in bad[:3]:
        print(f"  receipt {i}: {len(preds[i])} chars, ends with {preds[i][-80:]!r}")

    vals = [v for r in rows if parsed[r["id"]] for _, v in _raw_fields(parsed[r["id"]])]
    nums = sum(isinstance(v, (int, float)) for v in vals)
    print(f"values written as numbers instead of text: {nums} of {len(vals)}")

    strict = [doc_scores(r["gold"], parsed[r["id"]]) for r in rows]
    loose = [doc_scores(digits_only(r["gold"]), digits_only(parsed[r["id"]]) if parsed[r["id"]] else None)
             for r in rows]
    print(f"field F1, exact:                       {micro(strict):.3f}")
    print(f"field F1, prices and counts as digits: {micro(loose):.3f}")
    for sec in ["menu", "sub_total", "total"]:
        print(f"  {sec:10s} exact {micro(strict, sec):.3f}   digits {micro(loose, sec):.3f}")


def _raw_fields(obj):
    """Leaf values with their original types (fields() turns them into strings)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _raw_fields(v) if isinstance(v, (dict, list)) else [(k, v)]
    elif isinstance(obj, list):
        for v in obj:
            yield from _raw_fields(v)


if __name__ == "__main__":
    main(sys.argv[1])
