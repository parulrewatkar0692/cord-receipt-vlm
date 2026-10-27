#!/usr/bin/env python3
"""Score receipt parses against CORD v2 labels.

Field F1 as in the Donut paper: every receipt is flattened to (section.key, value) pairs, list
items share their key's path, and pairs are matched as a multiset with exact strings. Micro F1
over all receipts, with a bootstrap 95% range (1,000 resamples, seed 42). Also: a lenient F1
that ignores case and spaces, exact total_price accuracy, the valid-JSON rate and F1 per section.

Some validation and test receipts have exactly the same label as a train receipt (the same
receipt or order photographed again). overlap_ids() finds them so results can be reported with
and without them.

Usage
  python eval/score.py --self-test    # downloads the label column only; gold scores 1.0
"""
import json
import random
import re
from collections import Counter

LIST_KEYS = {"menu", "sub", "void_menu"}   # CORD stores a single item as a dict, several as a list
SECTIONS = ["menu", "void_menu", "sub_total", "total"]


def parse(text):
    """Model text -> (dict or None, strict). Strict means the text was only the JSON object,
    optionally inside a ```json code fence."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(t)
        return (obj, True) if isinstance(obj, dict) else (None, False)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return (obj, False) if isinstance(obj, dict) else (None, False)
        except json.JSONDecodeError:
            pass
    return None, False


def fields(obj, path=""):
    """Flatten to (path, value) pairs; list items share their key's path, as in Donut's field F1."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in LIST_KEYS and isinstance(v, dict):
                v = [v]
            yield from fields(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from fields(v, path)
    elif obj is not None and str(obj).strip():
        yield path, str(obj).strip()


def loose(v):
    return re.sub(r"\s+", "", v.lower())


def counts(obj, lenient=False):
    return Counter((p, loose(v) if lenient else v) for p, v in fields(obj))


def doc_scores(gold, pred, lenient=False):
    """{section: (tp, n_pred, n_gold)} plus "all"."""
    g, p = counts(gold, lenient), counts(pred or {}, lenient)
    out = {}
    for sec in SECTIONS + ["all"]:
        keep = (lambda k: True) if sec == "all" else (lambda k, s=sec: k[0].split(".")[0] == s)
        gs = Counter({k: n for k, n in g.items() if keep(k)})
        ps = Counter({k: n for k, n in p.items() if keep(k)})
        out[sec] = (sum((gs & ps).values()), sum(ps.values()), sum(gs.values()))
    return out


def f1(tp, n_pred, n_gold):
    if n_pred + n_gold == 0:
        return None
    return 2 * tp / (n_pred + n_gold)


def micro(docs, sec="all"):
    tp, npred, ngold = (sum(d[sec][i] for d in docs) for i in range(3))
    return f1(tp, npred, ngold)


def total_price(obj):
    t = (obj or {}).get("total")
    return str(t.get("total_price", "")).strip() if isinstance(t, dict) else ""


def score(golds, outputs, n_boot=1000, seed=42):
    parsed = [parse(o) for o in outputs]
    strict = [doc_scores(g, p) for g, (p, _) in zip(golds, parsed)]
    lenient = [doc_scores(g, p, lenient=True) for g, (p, _) in zip(golds, parsed)]
    rng = random.Random(seed)
    boot = sorted(micro([strict[rng.randrange(len(strict))] for _ in strict])
                  for _ in range(n_boot))
    has_total = [i for i, g in enumerate(golds) if total_price(g)]
    report = {
        "n": len(golds),
        "json_valid": sum(p is not None and s for p, s in parsed) / len(golds),
        "json_parsed": sum(p is not None for p, _ in parsed) / len(golds),
        "field_f1": micro(strict),
        "field_f1_ci95": [boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]],
        "field_f1_lenient": micro(lenient),
        "total_price_acc": sum(total_price(golds[i]) == total_price(parsed[i][0])
                               for i in has_total) / len(has_total),
        "by_section": {s: micro(strict, s) for s in SECTIONS},
    }
    return report, strict


def label_key(gold):
    return json.dumps(gold, sort_keys=True, ensure_ascii=False)


def overlap_ids(rows, train_golds):
    """Ids of rows whose label is identical to some train label."""
    train = {label_key(g) for g in train_golds}
    return {r["id"] for r in rows if label_key(r["gold"]) in train}


def load_labels(split, repo="naver-clova-ix/cord-v2"):
    """[{"id", "gold"}] for a split, reading only the label column (no images)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    rows = []
    for path in sorted(fs.glob(f"datasets/{repo}@~parquet/default/{split}/*.parquet")):
        with fs.open(path, block_size=1 << 20) as f:
            for s in pq.read_table(f, columns=["ground_truth"]).column("ground_truth").to_pylist():
                gt = json.loads(s)
                rows.append({"id": gt["meta"]["image_id"], "gold": gt["gt_parse"]})
    return rows


def self_test():
    val = load_labels("validation")
    golds = [r["gold"] for r in val]
    report, _ = score(golds, ["```json\n" + json.dumps(g) + "\n```" for g in golds])
    assert report["field_f1"] == 1.0 and report["json_valid"] == 1.0, report
    menu_dict = next(g for g in golds if isinstance(g.get("menu"), dict))
    assert doc_scores(menu_dict, dict(menu_dict, menu=[menu_dict["menu"]]))["all"][0] > 0
    empty, _ = score(golds, ["no receipt here"] * len(golds))
    assert empty["field_f1"] == 0.0 and empty["json_parsed"] == 0.0
    train = load_labels("train")
    test = load_labels("test")
    print(f"labels: train {len(train)}, validation {len(val)}, test {len(test)}")
    print(f"identical label in train: validation {len(overlap_ids(val, [r['gold'] for r in train]))}, "
          f"test {len(overlap_ids(test, [r['gold'] for r in train]))}")
    print("self-test passed")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        print(__doc__)
