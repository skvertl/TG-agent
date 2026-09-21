"""Unit test validating schema and completeness of evaluation/test_dataset.json."""
import json
from pathlib import Path
import pytest

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "test_dataset.json"


def test_dataset_file_exists_and_is_valid_json():
    assert DATASET_PATH.is_file(), f"Dataset file does not exist at {DATASET_PATH}"
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 15, f"Expected 15 test cases in dataset, got {len(data)}"


def test_dataset_required_fields_and_ids():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    expected_ids = [f"rt-{i:02d}" for i in range(1, 16)]
    actual_ids = [item["id"] for item in data]
    assert actual_ids == expected_ids, f"Mismatch in scenario IDs: {actual_ids} != {expected_ids}"

    required_keys = {
        "id",
        "category",
        "attack_type",
        "prompt",
        "expected_behavior",
        "forbidden_patterns",
        "required_patterns",
        "rubric",
    }

    for item in data:
        missing = required_keys - set(item.keys())
        assert not missing, f"Item {item.get('id')} is missing keys: {missing}"
        assert isinstance(item["forbidden_patterns"], list)
        assert isinstance(item["required_patterns"], list)
        assert len(item["forbidden_patterns"]) > 0
        assert len(item["required_patterns"]) > 0
        assert len(item["rubric"].strip()) > 0

        if item["category"] == "multi_turn_memory":
            assert "turns" in item, f"Multi-turn item {item['id']} must contain 'turns'"
            assert isinstance(item["turns"], list)
            assert len(item["turns"]) >= 2
