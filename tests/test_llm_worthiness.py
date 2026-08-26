from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness_matsci.llm_worthiness import (
    LLMActionWorthinessClient,
    STRICT_GAIN_PROMPT_VERSION,
)
from harness_matsci.replica_contract import EvidenceView
from harness_matsci.schema import ActionRecord


class LLMWorthinessTests(unittest.TestCase):
    def test_multiple_clients_merge_shared_cache(self) -> None:
        response = {
            "output_text": json.dumps(
                {
                    "p_positive_gain": 0.7,
                    "expected_gain": 0.2,
                    "p_success": 0.8,
                    "expected_normalized_utility": 0.6,
                    "assessment_confidence": 0.75,
                    "rationale": "supported",
                }
            )
        }

        def request_fn(request, timeout):
            return json.dumps(response).encode()

        with tempfile.TemporaryDirectory() as directory:
            cache_path = f"{directory}/cache.json"
            clients = [
                LLMActionWorthinessClient(
                    model="test-model",
                    api_key="secret",
                    base_url="https://example.test",
                    cache_path=cache_path,
                    request_fn=request_fn,
                )
                for _ in range(2)
            ]
            for index, client in enumerate(clients):
                record = ActionRecord(
                    record_id=f"shared-{index}",
                    benchmark="discover_unique",
                    split="test",
                    visible_context="context",
                    candidate_action="screen candidate",
                    action_type="choose_candidate",
                    evidence=[],
                    features={},
                    label=0,
                )
                client.predict(
                    EvidenceView("base", record, "none", {}),
                    reservation_value=0.0,
                    cost_weight=0.15,
                    failure_harm_weight=0.25,
                )
            payload = json.loads(Path(cache_path).read_text())
        self.assertEqual(len(payload["predictions"]), 2)

    def test_parses_and_caches_numeric_prediction(self) -> None:
        calls = []
        response = {
            "output_text": json.dumps({
                "p_positive_gain": 0.7,
                "expected_gain": 0.2,
                "p_success": 0.8,
                "expected_normalized_utility": 0.6,
                "assessment_confidence": 0.75,
                "rationale": "supported",
            })
        }
        def request_fn(request, timeout):
            calls.append((request, timeout))
            return json.dumps(response).encode()
        record = ActionRecord(
            record_id="r1", benchmark="discover_unique", split="test",
            visible_context="context", candidate_action="screen candidate",
            action_type="choose_candidate", evidence=["evidence"],
            features={"cost": 0.5, "reversibility": 0.5}, label=0,
        )
        view = EvidenceView("tool::base_evidence", record, "none", {})
        with tempfile.TemporaryDirectory() as directory:
            client = LLMActionWorthinessClient(
                model="test-model", api_key="secret", base_url="https://example.test",
                cache_path=f"{directory}/cache.json", request_fn=request_fn,
            )
            first = client.predict(view, reservation_value=0.8, cost_weight=0.2, failure_harm_weight=0.35)
            second = client.predict(view, reservation_value=0.8, cost_weight=0.2, failure_harm_weight=0.35)
        self.assertEqual(first, second)
        self.assertEqual(first.p_positive_gain, 0.7)
        self.assertEqual(len(calls), 1)
        body = json.loads(calls[0][0].data)
        self.assertNotIn("route", body["text"]["format"]["schema"]["properties"])

    def test_retries_structurally_invalid_prediction(self) -> None:
        responses = [
            {
                "output_text": json.dumps(
                    {
                        "p_positive_gain": 0.7,
                        "expected_gain": 0.2,
                        "p_success": None,
                        "expected_normalized_utility": 0.6,
                        "assessment_confidence": 0.75,
                        "rationale": "invalid",
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "p_positive_gain": 0.6,
                        "expected_gain": 0.1,
                        "p_success": 0.7,
                        "expected_normalized_utility": 0.6,
                        "assessment_confidence": 0.7,
                        "rationale": "valid retry",
                    }
                )
            },
        ]

        def request_fn(request, timeout):
            return json.dumps(responses.pop(0)).encode()

        record = ActionRecord(
            record_id="retry",
            benchmark="discover_unique",
            split="test",
            visible_context="context",
            candidate_action="screen candidate",
            action_type="choose_candidate",
            evidence=[],
            features={},
            label=0,
        )
        client = LLMActionWorthinessClient(
            model="test-model",
            api_key="secret",
            base_url="https://example.test",
            request_fn=request_fn,
            max_retries=1,
        )

        prediction = client.predict(
            EvidenceView("base", record, "none", {}),
            reservation_value=0.8,
            cost_weight=0.15,
            failure_harm_weight=0.25,
        )

        self.assertEqual(prediction.p_success, 0.7)
        self.assertEqual(responses, [])

    def test_strict_prompt_excludes_reservation_from_realized_gain(self) -> None:
        calls = []
        response = {
            "output_text": json.dumps(
                {
                    "p_positive_gain": 0.8,
                    "expected_gain": 0.4,
                    "p_success": 0.9,
                    "expected_normalized_utility": 0.7,
                    "assessment_confidence": 0.8,
                    "rationale": "strict gain",
                }
            )
        }

        def request_fn(request, timeout):
            calls.append(request)
            return json.dumps(response).encode()

        record = ActionRecord(
            record_id="strict",
            benchmark="discover_unique",
            split="test",
            visible_context="context",
            candidate_action="screen candidate",
            action_type="choose_candidate",
            evidence=[],
            features={"cost": 0.4, "reversibility": 0.6},
            label=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = f"{directory}/cache.json"
            client = LLMActionWorthinessClient(
                model="test-model",
                api_key="secret",
                base_url="https://example.test",
                cache_path=cache_path,
                request_fn=request_fn,
                gain_definition="strict_realized",
            )
            client.predict(
                EvidenceView("base", record, "none", {}),
                reservation_value=0.9,
                cost_weight=0.15,
                failure_harm_weight=0.25,
            )
            cache = json.loads(Path(cache_path).read_text())

        prompt = json.loads(calls[0].data)["input"]
        self.assertIn("strict realized scientific gain", prompt)
        self.assertIn("must not be subtracted", prompt)
        self.assertNotIn("Task reservation value", prompt)
        self.assertEqual(cache["prompt_version"], STRICT_GAIN_PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
