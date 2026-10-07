import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.services.query_processor import process_query


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "query_intent_eval.jsonl"
)


def load_cases(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Query evaluation dataset was not found: {path}")

    cases: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Query evaluation JSONL line {line_number} is invalid."
                ) from error
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("question"), str)
                or not record["question"].strip()
                or not isinstance(record.get("expected_intent"), str)
                or not record["expected_intent"].strip()
            ):
                raise ValueError(
                    f"Query evaluation line {line_number} must contain a "
                    "non-empty question and expected_intent."
                )
            cases.append(record)
    if not cases:
        raise ValueError("Query evaluation dataset is empty.")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the rule-based query processor against labeled questions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSONL file with question and expected_intent fields.",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    correct = 0
    totals: Counter[str] = Counter()
    correct_by_intent: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[tuple[str, str, str]] = []

    for case in cases:
        expected = case["expected_intent"]
        actual = process_query(case["question"]).intent
        totals[expected] += 1
        confusion[expected][actual] += 1
        if actual == expected:
            correct += 1
            correct_by_intent[expected] += 1
        else:
            errors.append((case["question"], expected, actual))

    print(f"Questions: {len(cases)}")
    print(f"Accuracy: {correct / len(cases):.1%} ({correct}/{len(cases)})")
    print("Per-intent accuracy:")
    for intent in sorted(totals):
        intent_total = totals[intent]
        intent_correct = correct_by_intent[intent]
        print(
            f"  {intent}: {intent_correct / intent_total:.1%} "
            f"({intent_correct}/{intent_total})"
        )

    if errors:
        print("Misclassified questions:")
        for question, expected, actual in errors:
            print(f"  {question!r}: expected {expected}, got {actual}")
    else:
        print("Misclassified questions: none")

    print("Confusion counts (expected -> actual):")
    for expected in sorted(confusion):
        counts = ", ".join(
            f"{actual}={count}"
            for actual, count in sorted(confusion[expected].items())
        )
        print(f"  {expected}: {counts}")


if __name__ == "__main__":
    main()
