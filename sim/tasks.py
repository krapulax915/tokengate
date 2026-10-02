"""Synthetic task generators with ground truth (spec section 12, phase 6).

Every generator returns a SimTask carrying the gateway payload (messages +
metadata.mock_expected / metadata.mock_wrong) and the checker that decides
success against the answer.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from sim.checkers import make_checker


@dataclass
class SimTask:
    task_id: str
    task_type: str
    messages: list[dict]
    checker_name: str
    expected: Any
    metadata: dict = field(default_factory=dict)

    def checker(self) -> Callable[[str], bool]:
        return make_checker(self.checker_name, self.expected)


_COMPANIES = ["Acme GmbH", "Nordwind AG", "Delta Works", "Kranich & Söhne", "Vega Trade Kft"]
_CURRENCIES = ["EUR", "USD", "GBP"]
_CATEGORIES = {
    "billing": ("invoice", "charge", "refund", "double", "payment", "subscription"),
    "shipping": ("delivery", "package", "tracking", "shipment", "late", "courier"),
    "quality": ("broken", "defect", "quality", "malfunction", "error message", "crash"),
}
_OTHER_MARKERS = ("newsletter", "partnership", "feedback about the website design")

_UPDATE_TOPICS = [
    ("deploy", "We deployed version 2.1 to production."),
    ("latency", "Median latency dropped by 30 percent after the fix."),
    ("billing", "The new billing system now supports annual plans."),
    ("security", "We rotated all access keys and enabled audit logs."),
]

_CODE_PROBLEMS = [
    ("add", "return the sum of a and b", "def add(a, b):\n    return a + b",
     "def add(a, b):\n    return a - b", r"def add\(a, b\):[\s\S]*return a \+ b\s*"),
    ("mul", "return the product of a and b", "def mul(a, b):\n    return a * b",
     "def mul(a, b):\n    return a / b", r"def mul\(a, b\):[\s\S]*return a \* b\s*"),
    ("neg", "return the negative of a", "def neg(a):\n    return -a",
     "def neg(a):\n    return a", r"def neg\(a\):[\s\S]*return -a\s*"),
]


def _gen_extract(rng: random.Random, task_id: str) -> SimTask:
    company = rng.choice(_COMPANIES)
    currency = rng.choice(_CURRENCIES)
    amount = round(rng.uniform(20, 5000), 2)
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    date = f"2026-{month:02d}-{day:02d}"
    expected = {"company": company, "date": date, "amount": amount, "currency": currency}
    wrong = dict(expected)
    if rng.random() < 0.5:
        wrong["amount"] = round(amount + rng.choice([1, -1, 10]), 2)
    else:
        wrong["company"] = rng.choice([c for c in _COMPANIES if c != company])
    return SimTask(
        task_id=task_id,
        task_type="extract",
        messages=[{
            "role": "user",
            "content": f"Extract the invoice fields from this text as JSON "
                       f"(company, date, amount, currency): Invoice from {company} "
                       f"dated {date}, total {amount} {currency}.",
        }],
        checker_name="json_schema",
        expected=expected,
        metadata={
            "mock_expected": json.dumps(expected, sort_keys=True),
            "mock_wrong": [json.dumps(wrong, sort_keys=True)],
        },
    )


def _gen_classify(rng: random.Random, task_id: str) -> SimTask:
    if rng.random() < 0.85:
        label = rng.choice(list(_CATEGORIES.keys()))
        keyword = rng.choice(_CATEGORIES[label])
        sentence = f"My {keyword} issue is that the {keyword} did not work as expected."
    else:
        label = "other"
        sentence = f"Please add me to the {rng.choice(_OTHER_MARKERS)} list."
    wrong_label = rng.choice([c for c in (*_CATEGORIES.keys(), "other") if c != label])
    return SimTask(
        task_id=task_id,
        task_type="classify",
        messages=[{
            "role": "user",
            "content": f"Classify the customer message into billing, shipping, quality, "
                       f"or other. Message: \"{sentence}\" Answer with the label only.",
        }],
        checker_name="exact",
        expected=label,
        metadata={"mock_expected": label, "mock_wrong": [wrong_label]},
    )


def _gen_summarize(rng: random.Random, task_id: str) -> SimTask:
    picked = rng.sample(_UPDATE_TOPICS, 4)
    keywords = [kw for kw, _ in picked]
    body = " ".join(s for _, s in picked)
    good = "Update summary: " + " ".join(kw for kw in keywords) + " were all mentioned."
    bad = f"Update summary: only {keywords[0]} and {keywords[1]} were mentioned."
    return SimTask(
        task_id=task_id,
        task_type="summarize",
        messages=[{
            "role": "user",
            "content": f"Summarize which topics appear in this update: {body} "
                       f"List every topic keyword.",
        }],
        checker_name="contains",
        expected=keywords,
        metadata={"mock_expected": good, "mock_wrong": [bad]},
    )


def _gen_reason(rng: random.Random, task_id: str) -> SimTask:
    pallets = rng.randint(2, 9)
    boxes = rng.randint(10, 60)
    damaged = rng.randint(1, min(30, pallets * boxes - 1))
    answer = pallets * boxes - damaged
    wrong = rng.choice([pallets * boxes, answer + 1, max(0, answer - 1)])
    return SimTask(
        task_id=task_id,
        task_type="reason",
        messages=[{
            "role": "user",
            "content": f"Calculate: a warehouse stores {pallets} pallets with {boxes} boxes "
                       f"each, and {damaged} boxes are damaged. How many good boxes remain? "
                       f"Answer with the number only.",
        }],
        checker_name="numeric",
        expected=str(answer),
        metadata={"mock_expected": str(answer), "mock_wrong": [str(wrong)]},
    )


def _gen_code(rng: random.Random, task_id: str) -> SimTask:
    name, desc, good, bad, pattern = rng.choice(_CODE_PROBLEMS)
    arg_list = ["a", "b"] if name != "neg" else ["a"]
    return SimTask(
        task_id=task_id,
        task_type="code",
        messages=[{
            "role": "user",
            "content": f"Write a Python function {name}({', '.join(arg_list)}) that {desc}. "
                       f"Answer with the function only.",
        }],
        checker_name="regex",
        expected=pattern,
        metadata={"mock_expected": good, "mock_wrong": [bad]},
    )


_GENERATORS: dict[str, Callable[[random.Random, str], SimTask]] = {
    "extract": _gen_extract,
    "classify": _gen_classify,
    "summarize": _gen_summarize,
    "reason": _gen_reason,
    "code": _gen_code,
}


def generate_task(task_type: str, rng: random.Random, task_id: str) -> SimTask:
    generator = _GENERATORS.get(task_type)
    if generator is None:
        raise ValueError(f"no generator for task type {task_type}")
    return generator(rng, task_id)


def generate_mix(mix: dict[str, float], rng: random.Random, count: int, prefix: str = "t") -> list[SimTask]:
    """Draw `count` tasks following the mix weights."""
    types = list(mix.keys())
    weights = [float(mix[t]) for t in types]
    population = rng.choices(types, weights=weights, k=count)
    return [generate_task(tt, rng, f"{prefix}-{i:05d}") for i, tt in enumerate(population)]
