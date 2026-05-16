import pytest

from forecasting.parser import extract_json_object, parse_prediction


def test_parse_prediction_from_json() -> None:
    payload = extract_json_object(
        'Here is the result:\n{"prediction": {"YES": 0.6, "NO": 0.4}, "rationale": "ok"}'
    )
    pred, rationale = parse_prediction(payload)
    assert abs(pred.YES - 0.6) < 1e-6
    assert rationale == "ok"


def test_parse_fenced_json() -> None:
    text = '```json\n{"prediction": {"YES": 0.7, "NO": 0.3}, "rationale": "x"}\n```'
    payload = extract_json_object(text)
    pred, _ = parse_prediction(payload)
    assert abs(pred.YES - 0.7) < 1e-6
