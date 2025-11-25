# PII Entity Recognition for Noisy STT Transcripts
**Kush Patil – Plivo Assignment (2025)**

## 1. Overview
This project builds a token-level NER system to detect PII entities inside noisy speech-to-text transcripts.  
The system predicts **BIO tags** using a learned token classifier and converts them into character-level spans with PII flags.

### Focus:
- **High precision** for PII categories (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE)
- **Fast CPU inference** (target p95 ≤ 20 ms)
- **Robust post-processing** and validation for noisy STT formats

### Supported Entity Types:
- `CREDIT_CARD` (PII)
- `PHONE` (PII)
- `EMAIL` (PII)
- `PERSON_NAME` (PII)
- `DATE` (PII)
- `CITY` (non-PII)
- `LOCATION` (non-PII)

---

## 2. Dataset

Synthetic STT-style datasets are created using:

```bash
python src/generate_stt_data.py --out_dir data --train_size 900 --dev_size 150
```

**This generates:**
- `data/train.jsonl`
- `data/dev.jsonl`

**Example entry:**
```json
{
  "id": "utt_0012",
  "text": "my credit card number ...",
  "entities": [
    { "start": 3, "end": 19, "label": "CREDIT_CARD" },
    { "start": 63, "end": 77, "label": "PERSON_NAME" },
    { "start": 81, "end": 105, "label": "EMAIL" }
  ]
}
```

---

## 3. Model Architecture

**Final model:** DistilBERT Token Classification Model

**Reasons:**
* **Strong accuracy** on noisy text
* **Excellent CPU latency** (6–15 ms)
* **Lightweight and stable** for PII extraction

The model is fine-tuned using:
- BIO tagging
- Focal loss
- Thresholded + validated span decoding

---

## 4. Training

**Fix torchvision import issue (required):**
```bash
export TRANSFORMERS_NO_TORCHVISION=1
```

**Train command (CPU):**
```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --device cpu \
  --epochs 3 \
  --batch_size 8
```

**Outputs:**
- `out/config.json`
- `out/pytorch_model.bin`
- Tokenizer files

---

## 5. Inference

**Predict on dev set:**
```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

**Output example:**
```json
{
  "id": "utt_0041",
  "entities": [
    { "start": 15, "end": 27, "label": "EMAIL", "pii": true },
    { "start": 43, "end": 50, "label": "DATE", "pii": true }
  ]
}
```

---

## 6. Evaluation

**Dev set:**
```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

**Stress set:**
```bash
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

---

## 7. Latency Measurement

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 100
```

---

## 8. Final Results (Kush Patil)

### Dev Set – Metrics

**Per-entity:**
```text
CREDIT_CARD     P=1.000 R=0.100 F1=0.182
DATE            P=1.000 R=0.348 F1=0.516
EMAIL           P=1.000 R=1.000 F1=1.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=1.000 R=1.000 F1=1.000
```

**Summary:**
* **Macro-F1:** 0.740
* **PII-only:**
    * Precision = 1.000
    * Recall    = 0.734
    * F1        = 0.846

**CPU Latency:**
* **p50:** 5.93 ms
* **p95:** 14.66 ms

### Stress Set – Metrics

**Per-entity:**
```text
CITY            P=0.000 R=0.000 F1=0.000
CREDIT_CARD     P=1.000 R=1.000 F1=1.000
DATE            P=1.000 R=0.613 F1=0.760
EMAIL           P=1.000 R=1.000 F1=1.000
LOCATION        P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=0.939 R=0.939 F1=0.939
```

**Summary:**
* **Macro-F1:** 0.671
* **PII-only:**
    * Precision = 0.984
    * Recall    = 0.899
    * F1        = 0.939

**CPU Latency:**
* **p50:** 6.66 ms
* **p95:** 10.44 ms

---

## 9. Key Design Choices

* **DistilBERT token classifier** for best accuracy-latency trade-off.
* **Focal loss** to help rare classes (especially CREDIT_CARD).
