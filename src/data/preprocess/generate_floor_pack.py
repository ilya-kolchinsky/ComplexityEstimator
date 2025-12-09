"""
Generate an extended Tiny Floor Pack (TFP) of trivially easy prompts.

Output CSV schema (one file per split):
    id,dataset,split,prompt_text,label_raw

- dataset: lowercased dataset name (default: "floorpack")
- label_raw: a stringified float in [0,1], already normalized
- split: train / dev / test

Usage:
    python generate_floor_pack.py --out_dir data/raw --n_train 6000 --n_dev 400 --n_test 400
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------------------
# RNG & small helpers
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def jitter(base: float, width: float = 0.015) -> float:
    return clip01(base + random.uniform(-width, width))


def is_trivial(prompt: str) -> bool:
    # Keep very short, single-turn, no code fences/URLs.
    if len(prompt.strip()) == 0:
        return True
    if len(prompt.split()) > 16:
        return False
    if "```" in prompt or "http" in prompt:
        return False
    return True


# ----------------------------
# Vocab pools for slots
# ----------------------------

LANGS = ["english", "spanish", "french", "german", "italian", "japanese", "korean", "portuguese"]
WORDS = ["ok", "ready", "hello", "thanks", "welcome", "yes", "no"]
SIMPLE_WORDS = ["cat", "dog", "apple", "big", "small", "car", "tree", "book", "happy", "cold", "warm", "chair", "city"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown", "gray"]
FRUITS = ["apple", "banana", "orange", "pear", "grape", "peach", "plum", "kiwi"]
ANIMALS = ["cat", "dog", "bird", "fish", "cow", "horse", "mouse", "lion"]
COUNTRIES = ["france", "spain", "germany", "italy", "japan", "brazil", "canada", "india"]
PROGLANGS = ["python", "java", "c", "javascript", "go", "rust", "ruby"]
SHAPES = ["circle", "square", "triangle", "rectangle", "oval", "star"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december"]
EMOJIS = ["👍", "👋", "😊", "🙂", "🤖", "✨", "🎉", "✅", "❤️"]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")

DATES = [
    "2020-01-01", "2019-12-25", "2021-07-04", "2018-11-11", "2000-02-29",
    "1999-01-01", "2010-05-15", "2015-10-10", "2005-06-30"
]

SMALL_NUMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TINY_AMOUNTS = [2, 3, 4, 5, 10, 12, 15, 20, 25, 30]


def render(template: str) -> str:
    t = template
    # Fill slots
    if "{lang}" in t:
        t = t.replace("{lang}", random.choice(LANGS))
    if "{word}" in t:
        t = t.replace("{word}", random.choice(WORDS))
    if "{simple_word}" in t:
        t = t.replace("{simple_word}", random.choice(SIMPLE_WORDS))
    if "{color}" in t:
        t = t.replace("{color}", random.choice(COLORS))
    if "{fruit}" in t:
        t = t.replace("{fruit}", random.choice(FRUITS))
    if "{animal}" in t:
        t = t.replace("{animal}", random.choice(ANIMALS))
    if "{country}" in t:
        t = t.replace("{country}", random.choice(COUNTRIES))
    if "{proglang}" in t:
        t = t.replace("{proglang}", random.choice(PROGLANGS))
    if "{shape}" in t:
        t = t.replace("{shape}", random.choice(SHAPES))
    if "{weekday}" in t:
        t = t.replace("{weekday}", random.choice(WEEKDAYS))
    if "{month}" in t:
        t = t.replace("{month}", random.choice(MONTHS))
    if "{emoji}" in t:
        t = t.replace("{emoji}", random.choice(EMOJIS))
    if "{letter}" in t:
        t = t.replace("{letter}", random.choice(LETTERS))
    if "{digit}" in t:
        t = t.replace("{digit}", random.choice(DIGITS))
    if "{n}" in t:
        t = t.replace("{n}", str(random.choice([2, 3, 4, 5])))
    if "{a}" in t:
        t = t.replace("{a}", str(random.choice(TINY_AMOUNTS)))
    if "{b}" in t:
        t = t.replace("{b}", str(random.choice(TINY_AMOUNTS)))
    if "{c}" in t:
        t = t.replace("{c}", str(random.choice(TINY_AMOUNTS)))
    if "{date}" in t:
        t = t.replace("{date}", random.choice(DATES))
    return t.strip()


# ----------------------------
# Buckets: many trivial patterns.
# base ∈ [0.00, 0.15] with jitter ±0.015
# ----------------------------

BUCKETS: List[Dict] = [
    # Greetings / identity / metainfo
    {"name": "greeting", "base": 0.02, "templates": [
        "hi", "hello", "hey", "good morning", "good evening", "hi there", "hello there", "yo", "howdy"
    ]},
    {"name": "identity", "base": 0.03, "templates": [
        "who are you?", "what are you?", "are you an ai?", "what can you do?", "how can you help me?"
    ]},
    {"name": "metainfo", "base": 0.05, "templates": [
        "explain how you work in one sentence.", "what is your purpose?", "describe yourself briefly."
    ]},
    {"name": "emoji_blank", "base": 0.01, "templates": EMOJIS + ["", "   ", "ok"]},

    # Formatting & echo
    {"name": "formatting", "base": 0.03, "templates": [
        "reply with 'ok'.", "respond with exactly 'YES'.", "output the word 'ready'.",
        "say hello in {lang}.", "write the word {word}.", "print 'done'."
    ]},
    {"name": "echo_paraphrase", "base": 0.05, "templates": [
        "repeat after me: {word}.", "rephrase: hello world.", "summarize in one word: {word}.",
        "uppercase: {word}.", "lowercase: {word}."
    ]},

    # Simple lists
    {"name": "simple_list", "base": 0.06, "templates": [
        "list {n} colors.", "list {n} fruits.", "name {n} animals.",
        "give {n} countries.", "list {n} programming languages.",
        "list {n} shapes.", "list {n} weekdays.", "list {n} months."
    ]},

    # Tiny define / synonym / antonym / spelling
    {"name": "tiny_define", "base": 0.07, "templates": [
        "define {simple_word}.", "what does {simple_word} mean?",
        "give a short definition of {simple_word}."
    ]},
    {"name": "tiny_synonym", "base": 0.07, "templates": [
        "synonym of {simple_word}.", "give one synonym for {simple_word}.",
        "what is a synonym for {simple_word}?"
    ]},
    {"name": "tiny_antonym", "base": 0.07, "templates": [
        "antonym of {simple_word}.", "give one opposite word for {simple_word}.",
        "what is an antonym for {simple_word}?"
    ]},
    {"name": "spelling", "base": 0.05, "templates": [
        "spell {simple_word}.", "how do you spell {simple_word}?"
    ]},

    # Colors / categories / set membership
    {"name": "category_check", "base": 0.04, "templates": [
        "is {color} a color?", "is {fruit} a fruit?", "is {animal} an animal?",
        "is {country} a country?", "is {proglang} a programming language?"
    ]},

    # Unit conversions (very small)
    {"name": "unit_convert_tiny", "base": 0.10, "templates": [
        "convert {a} km to m.", "convert {a} m to cm.", "convert {a} minutes to seconds.",
        "convert {a} hours to minutes.", "convert {a} kg to g."
    ]},

    # Static calendar / ordering
    {"name": "calendar_static", "base": 0.12, "templates": [
        "what day of week was {date}?", "what month does {date} fall in?"
    ]},
    {"name": "alphabet_ops", "base": 0.06, "templates": [
        "what letter comes after {letter}?", "what letter comes before {letter}?",
        "is {letter} a vowel?"
    ]},
    {"name": "ordering_short", "base": 0.06, "templates": [
        "alphabetize: dog, cat, ant.", "sort ascending: 3, 1, 2.",
        "sort words: {color}, {fruit}, {animal}."
    ]},

    # Yes/no basics & comparisons
    {"name": "yesno_basic", "base": 0.04, "templates": [
        "is water wet?", "is the sky blue?", "is {a} even?", "is {a} odd?",
        "is {a} greater than {b}?", "is {a} less than {b}?"
    ]},

    # Tiny arithmetic (no carry complexity)
    {"name": "tiny_arith", "base": 0.09, "templates": [
        "what is {a} + {b}?", "what is {a} - {b}?", "what is {a} × {b}?",
        "what is {a} * {b}?", "what is {a} + {b} + {c}?"
    ]},

    # Case / punctuation / whitespace
    {"name": "case_ops", "base": 0.03, "templates": [
        "capitalize: {simple_word}.", "title case: {simple_word} {simple_word}.",
        "remove spaces: h e l l o."
    ]},
    {"name": "punctuation_ops", "base": 0.03, "templates": [
        "add a period to the end of this: hello", "remove punctuation: hello!!!",
        "add a comma between these words: hello world"
    ]},

    # Simple pattern & sequence
    {"name": "sequence_next", "base": 0.08, "templates": [
        "what comes next: A B C ?", "what comes next: 1 2 3 ?",
        "fill the blank: monday, tuesday, _____"
    ]},

    # Short translation / romanization (one word)
    {"name": "simple_translate", "base": 0.06, "templates": [
        "say {word} in {lang}.", "translate {simple_word} to {lang}.",
        "give the {lang} word for {color}."
    ]},

    # Counting / length
    {"name": "counting", "base": 0.04, "templates": [
        "how many letters are in {simple_word}?", "how many words here: hello world?",
        "count the digits in 12345."
    ]},

    # Emphasis on exact output
    {"name": "exact_output", "base": 0.03, "templates": [
        "respond with exactly 'OK'.", "respond with exactly 'NO'.",
        "output exactly: {emoji}"
    ]},

    # Quote / bracket / wrap
    {"name": "wrapping", "base": 0.03, "templates": [
        "put quotes around {word}.", "wrap {word} in brackets.",
        "surround {word} with asterisks."
    ]},

    # Simple classification (binary, obvious)
    {"name": "binary_class_simple", "base": 0.05, "templates": [
        "is {animal} an animal? yes or no.", "is {proglang} a language for data science? yes or no."
    ]},

    # Choose first/last/shortest
    {"name": "pick_simple", "base": 0.05, "templates": [
        "choose the first word: cat dog bird.", "choose the last word: red blue green.",
        "which is shortest: {fruit}, {animal}, {color}?"
    ]},

    # Simple lower/upper bounds
    {"name": "minmax_simple", "base": 0.06, "templates": [
        "which is larger: {a} or {b}?", "which is smaller: {a} or {b}?"
    ]},

    # Shapes & colors affirmations
    {"name": "shape_color_affirm", "base": 0.04, "templates": [
        "is {shape} a shape?", "is {color} a color?"
    ]},

    # Months / weekdays trivia (static)
    {"name": "month_weekday_facts", "base": 0.06, "templates": [
        "is {month} a month?", "is {weekday} a weekday?"
    ]},

    # Simple placeholders with numbers/letters
    {"name": "echo_number_letter", "base": 0.03, "templates": [
        "repeat the number {a}.", "say the letter {letter}.", "output the digit {digit}."
    ]},
]


# ----------------------------
# Allocation & sampling
# ----------------------------

def default_weights() -> Dict[str, float]:
    w = {b["name"]: 1.0 for b in BUCKETS}
    # Slight reweights for variety
    for k in ["greeting", "identity", "formatting", "simple_list", "tiny_define", "tiny_synonym", "yesno_basic"]:
        w[k] = 1.3
    for k in ["emoji_blank", "calendar_static"]:
        w[k] = 0.7
    return w


def allocate_counts(n_total: int, weights: Dict[str, float]) -> Dict[str, int]:
    names = list(weights.keys())
    raw = [weights[n] for n in names]
    s = sum(raw)
    props = [w / s for w in raw]
    counts = [int(round(n_total * p)) for p in props]
    diff = n_total - sum(counts)
    # Adjust by largest remainder
    rema = sorted(
        [(i, props[i] - counts[i] / max(n_total, 1)) for i in range(len(counts))],
        key=lambda x: x[1],
        reverse=(diff > 0)
    )
    i = 0
    while diff != 0 and i < len(rema):
        idx = rema[i][0]
        counts[idx] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    return {names[i]: max(0, counts[i]) for i in range(len(names))}


def gen_bucket_samples(bucket: Dict, k: int) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    seen = set()
    attempts = 0
    max_attempts = k * 15 + 50
    while len(out) < k and attempts < max_attempts:
        attempts += 1
        tmpl = random.choice(bucket["templates"])
        prompt = render(tmpl)
        if not is_trivial(prompt):
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        score = jitter(bucket["base"], 0.015)
        out.append((prompt, score))
    return out


def build_split(n_items: int) -> List[Tuple[str, float, str]]:
    weights = default_weights()
    alloc = allocate_counts(n_items, weights)
    rows: List[Tuple[str, float, str]] = []
    for b in BUCKETS:
        k = alloc[b["name"]]
        if k <= 0:
            continue
        rows.extend((p, s, b["name"]) for (p, s) in gen_bucket_samples(b, k))
    random.shuffle(rows)
    return rows


def write_csv(rows: List[Tuple[str, float, str]], split: str, dataset_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "split", "prompt_text", "label_raw"])
        for i, (prompt, score, bucket) in enumerate(rows):
            w.writerow([f"{dataset_name}:{split}:{i}", dataset_name, split, prompt, f"{score:.3f}"])
    return path


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/floorpack")
    ap.add_argument("--dataset_name", type=str, default="floorpack")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_dev", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)

    train_rows = build_split(args.n_train)
    dev_rows = build_split(args.n_dev)
    test_rows = build_split(args.n_test)

    p_train = write_csv(train_rows, "train", args.dataset_name, out_dir)
    p_dev = write_csv(dev_rows, "dev", args.dataset_name, out_dir)
    p_test = write_csv(test_rows, "test", args.dataset_name, out_dir)

    print("Wrote:")
    print(" -", p_train)
    print(" -", p_dev)
    print(" -", p_test)


if __name__ == "__main__":
    main()
