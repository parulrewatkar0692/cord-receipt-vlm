# CORD receipts: vision-language fine-tuning probe

Can a small vision-language model (VLM) read a receipt photo and return its items, prices and
totals as JSON, and how much would fine-tuning help? This repository is an exploration next to
the main capstone ([finance-complaint-triage](https://github.com/parulrewatkar0692/finance-complaint-triage)),
to decide whether a VLM project on CORD is worth switching to.

## Data

[CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Consolidated Receipt Dataset,
NAVER Clova), licence **CC-BY-4.0**. Photos of Indonesian restaurant receipts, each labelled by
the dataset authors with a JSON parse:

```json
{"menu": [{"nm": "Ice Lemon Tea", "cnt": "1 x", "price": "24,000"}],
 "sub_total": {"subtotal_price": "24,000", "tax_price": "2,400"},
 "total": {"total_price": "26,400", "cashprice": "50,000", "changeprice": "23,600"}}
```

| Split | Receipts | Label identical to a train receipt |
| --- | --- | --- |
| train | 800 | 43 duplicates within train |
| validation | 100 | 14 |
| test | 100 | 10 |

The identical labels are most likely the same receipt or order photographed again. Results are
reported with and without them. Test stays closed while exploring; all numbers here are on
validation.

Citation: Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, Document
Intelligence Workshop at NeurIPS 2019.

## Zero-shot probe

[`notebooks/cord_vlm_probe_colab.ipynb`](notebooks/cord_vlm_probe_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/parulrewatkar0692/cord-receipt-vlm/blob/main/notebooks/cord_vlm_probe_colab.ipynb))
runs `Qwen/Qwen2.5-VL-3B-Instruct` with no fine-tuning on the 100 validation receipts (T4 GPU,
greedy decoding, images capped at about 800k pixels) and scores it with
[`eval/score.py`](eval/score.py):

- **field F1**: Donut-style; every `section.key = value` pair, exact string match, micro F1 with
  a bootstrap 95% range
- **field F1, lenient**: the same, ignoring case and spaces
- **total_price accuracy**: exact match of the receipt total
- **valid JSON**: the answer is one JSON object (a ```json fence is allowed)

`python eval/score.py --self-test` checks the scorer on the real labels (gold scores 1.0) and
prints the overlap counts. It needs `pyarrow` and `huggingface_hub` and reads only the label
column, not the images.

### Results (2026-10-27, Colab T4)

| Metric | All (n=100) | Without train overlap (n=86) |
| --- | --- | --- |
| Valid JSON | 0.990 | 0.988 |
| **Field F1** (exact) | **0.483** [0.438, 0.523] | 0.475 [0.430, 0.521] |
| Field F1, lenient (case and spaces) | 0.492 | 0.485 |
| total_price exact | 0.745 | 0.726 |
| F1 menu / sub_total / total | 0.486 / 0.448 / 0.493 | 0.475 / 0.444 / 0.488 |

84 receipts were answered in float16 and 16 in bfloat16 (see below); 21.6 seconds per receipt on
average.

**float16 failures and the retry.** In the first, float16-only run, 17 answers were 1,024
characters of `!!!!`: a numeric overflow of Qwen2.5-VL in float16, not a reading error. That run
scored field F1 0.453 with 83% valid JSON. The notebook now retries such answers in bfloat16
(emulated on the T4, still about 20 seconds per receipt): all 16 retried receipts gave a real
answer. The one remaining invalid answer was cut off at the token limit, not garbage.

**What else holds the score down** (`python eval/diagnose.py <predictions>`, first run): writing
numbers instead of the printed text (`16500` for `"16,500"`) affected 256 of 3,024 values;
comparing prices and counts as digits only raised field F1 from 0.453 to 0.524.

So the zero-shot 3B model is far below the published fine-tuned results, which leaves clear room
for fine-tuning to show a gain; copying values exactly as printed is one of the things it should
teach.

For reference, published fine-tuned results on CORD are roughly 0.84 field F1 for Donut
(OCR-free) up to about 0.96 for OCR-based models such as LayoutLMv3.
