#!/usr/bin/env python3
"""
Generate synthetic noisy STT train/dev JSONL for PII NER.
Usage:
  python scripts/generate_stt_data.py --out_dir data --train_size 900 --dev_size 150 --seed 42
"""
import json
import random
import argparse
import os
from datetime import datetime

NAMES = [
    "aditya rao","neha dubey","sanjay krishnan","deepak krishnan","ramesh singh",
    "tushar dubey","ravi dubey","sanjay patel","ananya krishnan","meera patel",
    "vikram agarwal","pooja mehta","rohan patel","ananya pillai","yash joshi",
    "soumya reddy","kiran mehta","tanya chaudhary","arjun dubey","varun singh"
]

CITIES = [
    "mumbai","delhi","bangalore","hyderabad","chennai","kolkata","jaipur","lucknow",
    "pune","surat","nagpur","coimbatore","trivandrum","noida","gurgaon"
]

LOCATIONS = [
    "old airport road","hitech city","koramangala","mg road","outer ring road",
    "banjara hills","salt lake","mall road","velachery","electronic city"
]

EMAIL_DOMAINS = ["gmail.com","hotmail.com","yahoo.com","rediffmail.com","protonmail.com","outlook.com"]

FILLERS = ["uh","haan","so","actually","please","i am","we can meet","my phone is","email is"]

# Helpers for spoken forms
NUM_WORDS = {
    '0': ['zero','oh','o'],
    '1': ['one'],
    '2': ['two'],
    '3': ['three'],
    '4': ['four'],
    '5': ['five'],
    '6': ['six'],
    '7': ['seven'],
    '8': ['eight'],
    '9': ['nine'],
}

def digits_to_spoken(digits, allow_double=True, word_sep=" "):
    """Turn a digit string into a spoken form with occasional 'double' or spelled digits."""
    out = []
    i = 0
    while i < len(digits):
        d = digits[i]
        # make doubles sometimes
        if allow_double and i+1 < len(digits) and digits[i+1] == d and random.random() < 0.4:
            out.append("double " + random.choice(NUM_WORDS[d]))
            i += 2
            continue
        # grouped numbers vs single
        if random.random() < 0.6:
            out.append(random.choice(NUM_WORDS[d]))
        else:
            out.append(d)  # keep numeric
        i += 1
    return word_sep.join(out)

def format_credit_card(use_spoken=True):
    # Create 16-digit credit card, sometimes with spaces/hyphens, sometimes as spoken words
    digits = "".join(str(random.randint(0,9)) for _ in range(16))
    if use_spoken and random.random() < 0.6:
        # grouped in 4s or full spoken
        pieces = [digits[i:i+4] for i in range(0,16,4)]
        if random.random() < 0.7:
            # mixed numeric and spoken
            return " ".join(random.choice([p, digits_to_spoken(p)]) for p in pieces), digits
        else:
            # fully spoken
            return " ".join(digits_to_spoken(p) for p in pieces), digits
    else:
        sep = random.choice([" ", "-", ""])
        if sep == "":
            return digits, digits
        return sep.join(digits[i:i+4] for i in range(0,16,4)), digits

def format_phone(use_spoken=True):
    # Typical 10-digit Indian phone
    digits = "".join(str(random.randint(0,9)) for _ in range(10))
    if use_spoken and random.random() < 0.7:
        # group as 5+5 or 3+3+4 etc, as spoken or numeric
        if random.random() < 0.6:
            groups = [digits[:5], digits[5:]]
        else:
            groups = [digits[:3], digits[3:6], digits[6:]]
        return " ".join(digits_to_spoken(g) if random.random() < 0.6 else g for g in groups), digits
    else:
        sep = random.choice([" ", ""])
        if sep == "":
            return digits, digits
        # sometimes spaced 5+5
        if random.random() < 0.5:
            return digits[:5] + " " + digits[5:], digits
        return digits, digits

def format_email(name=None):
    # variations: dot, spelled domains, spaces between letters for obfuscation
    if name is None:
        name = random.choice(NAMES)
    # pick representation of name
    local = random.choice([name.replace(" ", "."), name.replace(" ", ""), name.split()[0]+"."+name.split()[-1]])
    domain = random.choice(EMAIL_DOMAINS)
    # sometimes use spoken 'at' 'dot' and sometimes real
    if random.random() < 0.6:
        local_rep = local.replace(".", " dot ")
        domain_rep = domain.replace(".", " dot ")
        return f"{local_rep} at {domain_rep}", f"{local}@{domain}"
    # mix with spaced letters (like g m a i l)
    if random.random() < 0.15:
        dom = domain.split(".")[0]
        dom_spaced = " ".join(list(dom)) + " dot " + domain.split(".")[1]
        return f"{local} at {dom_spaced}", f"{local}@{domain}"
    return f"{local}@{domain}", f"{local}@{domain}"

def format_date():
    # variations of dates seen in dataset
    day = random.randint(1,28)
    month = random.choice(["january","february","march","april","may","june","july","august","september","october","november","december","01","02","03","04","05","06","07","08","09","10","11","12"])
    year = random.choice(["2023","2024","2025","2026"])
    if random.random() < 0.4:
        # numeric dd/mm/yyyy or dd-mm-yyyy
        sep = random.choice(["/","-"])
        return f"{day:02d}{sep}{random.randint(1,12):02d}{sep}{random.choice([year, '2023'])}"
    if random.random() < 0.6:
        # "5 july 2023" or "5 of july 2023"
        return f"{day} {month} {year}" if random.random()<0.6 else f"{day} of {month} {year}"
    return f"{day} {month} {year}"

def random_prefix():
    parts = []
    if random.random() < 0.5:
        parts.append(random.choice(["hi","hello","hey","haan","um","uh","so"]))
    if random.random() < 0.2:
        parts.append(random.choice(["this is","my name is","i am","i am"]))
    return " ".join(parts).strip()

def assemble_uttr(person=None, include=None):
    # Build a single utterance text and entity list
    # include: list of entity types to include in this utterance
    if person is None:
        person = random.choice(NAMES)
    if include is None:
        include = []
    segments = []
    entities = []

    # optional prefix
    pref = random_prefix()
    if pref:
        segments.append(pref)

    # If PERSON_NAME
    if "PERSON_NAME" in include:
        s = " ".join([random.choice(["this is","my name is","i am"]), person])
        start = sum(len(x)+1 for x in segments) if segments else 0
        if segments: start -= 1  # adjust for not adding extra space at front
        # append with correct spacing
        if segments:
            segments.append(s)
        else:
            segments = [s]
        # compute text so far
        text_so_far = " ".join(segments)
        name_start = text_so_far.index(person)
        name_end = name_start + len(person)
        # record; we will recompute final offsets later (safer to rebuild entire text at end)
        entities.append(("PERSON_NAME", person, name_start, name_end))

    # CITY
    if "CITY" in include:
        city = random.choice(CITIES)
        seg = f"from {city}" if random.random() < 0.6 else f"traveling to {city}"
        segments.append(seg)
        # mark later

        # get approximate offsets later

    # LOCATION
    if "LOCATION" in include:
        loc = random.choice(LOCATIONS)
        seg = f"in {loc}"
        segments.append(seg)

    # PHONE
    if "PHONE" in include:
        phone_text, phone_digits = format_phone()
        seg = f"my phone is {phone_text}"
        segments.append(seg)
        entities.append(("PHONE", phone_text, None, None))  # offsets later

    # EMAIL
    if "EMAIL" in include:
        email_text, email_norm = format_email(person)
        seg = f"and email is {email_text}" if random.random() < 0.8 else f"email id of {person} is {email_text}"
        segments.append(seg)
        entities.append(("EMAIL", email_text, None, None))

    # CREDIT_CARD
    if "CREDIT_CARD" in include:
        cc_text, cc_digits = format_credit_card()
        seg = f"my credit card number is {cc_text} and it expires on {format_date()}"
        segments.append(seg)
        entities.append(("CREDIT_CARD", cc_text, None, None))
        # also add DATE entity for expiry
        entities.append(("DATE", "", None, None))

    # DATE (meeting)
    if "DATE" in include and "CREDIT_CARD" not in include:
        date_text = format_date()
        seg = f"we can meet on {date_text}"
        segments.append(seg)
        entities.append(("DATE", date_text, None, None))

    # filler tail
    if random.random() < 0.3:
        segments.append(random.choice(["please call me tomorrow","please","thanks","ok","bye"]))

    text = " ".join(segments).strip()

    # compute character offsets for each recorded entity reliably
    spans = []
    # find occurrences of the entity text in the assembled string (first match)
    for e in entities:
        lab, ent_text, s_off, e_off = e
        if lab == "PERSON_NAME":
            # use person string
            idx = text.find(person)
            if idx >= 0:
                spans.append({"start": idx, "end": idx+len(person), "label": lab})
        elif lab == "DATE" and ent_text:
            idx = text.find(ent_text)
            if idx >= 0:
                spans.append({"start": idx, "end": idx+len(ent_text), "label": "DATE"})
        elif lab == "DATE" and not ent_text:
            # expiry date in the previously added segment; try to find a numeric date pattern
            # crude: find last occurrence of a 4-digit year nearby
            possible_idxs = []
            for y in ["2023","2024","2025","2026"]:
                p = text.find(y)
                if p >= 0:
                    possible_idxs.append(p)
            if possible_idxs:
                yidx = possible_idxs[-1]
                # expand left to find start of that token
                l = yidx
                while l>0 and text[l-1] not in (" ","/","-"):
                    l-=1
                spans.append({"start": l, "end": yidx+4, "label": "DATE"})
        else:
            idx = text.find(ent_text)
            if idx >= 0:
                spans.append({"start": idx, "end": idx+len(ent_text), "label": lab})

    return text, spans

def generate_dataset(out_path, n_samples, seed=42, pii_bias=0.8):
    random.seed(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(1, n_samples+1):
            uid = f"utt_{i:04d}"
            # decide which PII to include; bias to include PII more often for precision-focused set
            include = []
            if random.random() < pii_bias:  # include at least one PII entity most of the time
                # prefer PERSON / PHONE / EMAIL / CREDIT_CARD / DATE
                if random.random() < 0.7:
                    include.append("PERSON_NAME")
                if random.random() < 0.5:
                    include.append("PHONE")
                if random.random() < 0.35:
                    include.append("EMAIL")
                if random.random() < 0.25:
                    include.append("CREDIT_CARD")
                if random.random() < 0.35:
                    include.append("DATE")
            else:
                # non-PII examples: city/location or no entities
                if random.random() < 0.6:
                    include.append(random.choice(["CITY","LOCATION"]))
            # ensure at least something realistic
            text, spans = assemble_uttr(person=random.choice(NAMES), include=include)
            obj = {"id": uid, "text": text, "entities": spans}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--train_size", type=int, default=900)
    ap.add_argument("--dev_size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = os.path.join(args.out_dir, "train.jsonl")
    dev_path = os.path.join(args.out_dir, "dev.jsonl")

    print("Generating train ->", train_path)
    generate_dataset(train_path, args.train_size, seed=args.seed, pii_bias=0.85)
    print("Generating dev ->", dev_path)
    generate_dataset(dev_path, args.dev_size, seed=args.seed+1, pii_bias=0.90)
    print("Done.")

if __name__ == "__main__":
    main()
