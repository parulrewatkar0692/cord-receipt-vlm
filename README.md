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

### Results

Not run yet.

For reference, published fine-tuned results on CORD are roughly 0.84 field F1 for Donut
(OCR-free) up to about 0.96 for OCR-based models such as LayoutLMv3.
