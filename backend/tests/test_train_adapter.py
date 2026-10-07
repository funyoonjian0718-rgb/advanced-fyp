import json
from pathlib import Path

import pytest

from app.ml.train_adapter import _read_examples
from app.services.generation import _has_reviewed_adapter


def test_loader_accepts_human_reviewed_v2_messages(tmp_path: Path) -> None:
    dataset = tmp_path / "curated.jsonl"
    records = [
        {
            "reviewed": True,
            "question": f"Question {index}",
            "answer": f"Reviewed answer {index}",
        }
        for index in range(3)
    ]
    records.append(
        {
            "reviewed": True,
            "messages": [
                {"role": "system", "content": "Be cautious."},
                {"role": "user", "content": "What does this label say?"},
                {"role": "assistant", "content": "It says ..."},
            ],
        }
    )
    dataset.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    examples = _read_examples(dataset)

    assert len(examples) == 4
    assert examples[-1].messages[-1] == {
        "role": "assistant",
        "content": "It says ...",
    }


def test_loader_rejects_examples_not_marked_human_reviewed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "unreviewed.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Question", "response": "Answer"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed=true"):
        _read_examples(dataset)


def test_adapter_requires_reviewed_training_manifest(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert _has_reviewed_adapter(tmp_path) is False

    (tmp_path / "curated_training_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "human_reviewed_examples": 4,
            }
        ),
        encoding="utf-8",
    )
    assert _has_reviewed_adapter(tmp_path) is True
