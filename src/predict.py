# src/predict.py
import json
import argparse
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "double": "DOUBLE", "triple": "TRIPLE"
}

EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
PHONE_DIGIT_RE = re.compile(r"\d")
CREDIT_MIN_DIGITS = 13
CREDIT_MAX_DIGITS = 19


def normalize_spoken_digits(s: str) -> str:
    """
    Convert spoken forms to digits, handle 'double' -> repeat previous digit.
    This is a simple normalizer used for validation only.
    """
    tokens = s.strip().split()
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t == "double" and out:
            # repeat previous digit once more
            out.append(out[-1])
            i += 1
            continue
        if t in DIGIT_WORDS and DIGIT_WORDS[t] != "DOUBLE":
            out.append(DIGIT_WORDS[t])
            i += 1
            continue
        # try if token contains digits already
        if any(ch.isdigit() for ch in t):
            # keep digits only
            digits = "".join([ch for ch in t if ch.isdigit()])
            out.extend(list(digits))
            i += 1
            continue
        # spelled-out letters (like h o t m a i l) - treat as not digits here
        i += 1
    return "".join(out)


def normalize_email_text(s: str) -> str:
    # Convert "dot" -> ".", "at" -> "@", collapse spaced letters like "g m a i l"
    s = s.strip()
    # collapse spaced letters (e.g., "g m a i l")
    s = re.sub(r"\b([a-z])(?:\s+[a-z]){1,}\b", lambda m: m.group(0).replace(" ", ""), s, flags=re.IGNORECASE)
    s = s.replace(" dot ", ".").replace(" dot", ".").replace("dot ", ".")
    s = s.replace(" at ", "@").replace(" at", "@").replace("at ", "@")
    s = s.replace(" ", "")
    # multiple dots collapse
    s = re.sub(r"\.{2,}", ".", s)
    return s


def luhn_check(card_num: str) -> bool:
    s = card_num[::-1]
    total = 0
    for i, ch in enumerate(s):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def bio_to_spans_with_confidence(offsets, label_ids, probs):
    """
    offsets: list of (start,end)
    label_ids: list of predicted label ids (ints)
    probs: list of arrays or lists representing softmax probabilities per token (len = len(offsets), each is vector length num_labels)
    Returns list of (start, end, label_str, mean_confidence)
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None
    confidences = []

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if start == end:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                mean_conf = float(sum(confidences) / max(1, len(confidences)))
                spans.append((current_start, current_end, current_label, mean_conf))
                current_label = None
                confidences = []
            continue

        prefix, ent_type = label.split("-", 1)
        token_conf = float(probs[idx][lid])
        if prefix == "B":
            if current_label is not None:
                mean_conf = float(sum(confidences) / max(1, len(confidences)))
                spans.append((current_start, current_end, current_label, mean_conf))
            current_label = ent_type
            current_start = start
            current_end = end
            confidences = [token_conf]
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                confidences.append(token_conf)
            else:
                # New entity without B: start new
                if current_label is not None:
                    mean_conf = float(sum(confidences) / max(1, len(confidences)))
                    spans.append((current_start, current_end, current_label, mean_conf))
                current_label = ent_type
                current_start = start
                current_end = end
                confidences = [token_conf]

    if current_label is not None:
        mean_conf = float(sum(confidences) / max(1, len(confidences)))
        spans.append((current_start, current_end, current_label, mean_conf))

    return spans


# per-entity minimum confidence threshold (tuneable)
DEFAULT_THRESHOLDS = {
    "EMAIL": 0.90,
    "CREDIT_CARD": 0.85,
    "PHONE": 0.85,
    "PERSON_NAME": 0.75,
    "DATE": 0.70,
    "CITY": 0.30,
    "LOCATION": 0.30,
}


def validate_and_filter_span(text: str, s: int, e: int, label: str, conf: float) -> bool:
    span_text = text[s:e].strip()
    if label == "EMAIL":
        norm = normalize_email_text(span_text)
        return bool(EMAIL_RE.match(norm))
    if label == "PHONE":
        digits = normalize_spoken_digits(span_text)
        # If not enough digits from spoken normalization, fallback to extracting digits
        if len(digits) < 7:
            digits = "".join(PHONE_DIGIT_RE.findall(span_text))
        return 7 <= len(digits) <= 15
    if label == "CREDIT_CARD":
        digits = "".join(PHONE_DIGIT_RE.findall(span_text))
        # If digits are spelled-out, try spoken conversion
        if len(digits) < CREDIT_MIN_DIGITS:
            spoken = normalize_spoken_digits(span_text)
            if len(spoken) > len(digits):
                digits = spoken
        if not (CREDIT_MIN_DIGITS <= len(digits) <= CREDIT_MAX_DIGITS):
            return False
        # Luhn check (optional but increases precision)
        try:
            return luhn_check(digits)
        except Exception:
            return False
    if label == "DATE":
        # Very permissive date validator: look for month name or digits/slash/dash
        month_names = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        low = span_text.lower()
        if any(m in low for m in month_names):
            return True
        # slashes or dashes or numeric patterns
        if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", span_text):
            return True
        if re.search(r"\b\d{1,2}\s+(of\s+)?[a-zA-Z]{3,}\b", span_text):
            return True
        return False
    if label == "PERSON_NAME":
        # Reject if digits in name
        if re.search(r"\d", span_text):
            return False
        # allow 1-4 tokens name
        toks = span_text.split()
        if len(toks) > 4 or len(toks) < 1:
            return False
        # require at least one alphabetic token > 1 char
        if not any(len(t) > 1 and t.isalpha() for t in toks):
            return False
        # optionally accept only if confidence high (handled by thresholds)
        return True
    # CITY/LOCATION: accept, no validation
    return True


def remove_overlaps_keep_highest(spans):
    """
    spans: list of dicts with s,e,label,conf
    Remove overlaps by simple greedy: sort by conf desc, keep if not overlapping with already kept
    """
    kept = []
    occupied = []
    for sp in sorted(spans, key=lambda x: x["conf"], reverse=True):
        s, e = sp["start"], sp["end"]
        overlap = False
        for (a, b) in occupied:
            if not (e <= a or s >= b):
                overlap = True
                break
        if not overlap:
            kept.append(sp)
            occupied.append((s, e))
    # restore original order by start
    kept = sorted(kept, key=lambda x: x["start"])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thresholds", default=None, help="JSON string or path with per-label thresholds")
    args = ap.parse_args()

    # optional thresholds override
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.thresholds:
        try:
            if os.path.exists(args.thresholds):
                with open(args.thresholds, "r") as fh:
                    thresholds.update(json.load(fh))
            else:
                thresholds.update(json.loads(args.thresholds))
        except Exception:
            print("Could not parse thresholds argument; using defaults.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # (L, C)
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            raw_spans = bio_to_spans_with_confidence(offsets, pred_ids, probs)

            candidate_spans = []
            for (s, e, lab, conf) in raw_spans:
                # apply per-label confidence threshold
                thr = float(thresholds.get(lab, 0.3))
                if conf < thr:
                    # skip if below threshold
                    continue
                # run validator (regex / numeric checks)
                if not validate_and_filter_span(text, s, e, lab, conf):
                    continue
                candidate_spans.append({"start": int(s), "end": int(e), "label": lab, "conf": float(conf), "pii": bool(label_is_pii(lab))})

            # remove overlaps keeping highest confidence
            final_spans = remove_overlaps_keep_highest(candidate_spans)

            # strip conf from output
            ents = [{"start": sp["start"], "end": sp["end"], "label": sp["label"], "pii": sp["pii"]} for sp in final_spans]
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
